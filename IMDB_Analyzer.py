from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import requests
from imdb import Cinemagoer, IMDbError
from dotenv import load_dotenv
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import re

SERIES_NOT_FOUN_ERROR = -1

# --------------------------------------------------------------------------- #
#  Environment Variables
# --------------------------------------------------------------------------- #
load_dotenv()  # loads from .env by default

OMDB_API_KEY = os.getenv("OMDB_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OPENSUBTITLES_API_KEY = os.getenv("OPENSUBTITLES_API_KEY")

# --------------------------------------------------------------------------- #
#  Logging configuration
# --------------------------------------------------------------------------- #
_LOG_PATH = Path(__file__).with_suffix(".log")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(),  # print()‑like console output
        logging.FileHandler(_LOG_PATH, encoding="utf‑8")
    ],
)
log = logging.getLogger("CinemagoerClient")


class Person:
    """Lightweight domain object representing a person on IMDb."""

    def __init__(self, person_dict: Dict[str, Any]) -> None:
        self.person_id: str = person_dict.get("personID")
        self.name: str = person_dict.get("name")
        self.roles: List[str] = person_dict.get("roles", [])

    def __repr__(self) -> str:
        return f"<Person {self.name!r} ({self.person_id})>"


class Title:
    """Domain object for a movie / series / episode."""

    def __init__(self, movie_dict: Dict[str, Any]) -> None:
        self.movie_id: str = movie_dict.get("movieID")
        self.title: str = movie_dict.get("title")
        self.kind: str = movie_dict.get("kind")
        self.year: int | None = movie_dict.get("year")
        self.rating: float | None = movie_dict.get("rating")
        self.cast: List[Person] = [
            Person(p) for p in movie_dict.get("cast", [])
        ]
        self.episodes: List = movie_dict.get("episodes", [])

    def __repr__(self) -> str:
        return f"<Title {self.title!r} ({self.year}) – {self.kind}>"

    def top_cast(self, n: int = 5) -> List[str]:
        """Return a list with names of the top‑billed cast."""
        return [person.name for person in self.cast[:n]]


class CinemagoerClient:
    """
    High‑level client providing *typed*, *logged* helpers around Cinemagoer.
    """

    def __init__(self, language: str | None = None) -> None:
        self.ia = Cinemagoer()
        if language:
            self.ia.set_locale(language)
            log.info("Locale set to %s", language)

    def get_top_tv_shows(self, top_n: int = 100):
        top_n_tv_shows_path = f"Data/top_{top_n}_tv_shows.json"

        if os.path.exists(top_n_tv_shows_path):
            log.info(f"Load existing {top_n_tv_shows_path}")
            with open(top_n_tv_shows_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            log.info(f"Fetching top {top_n} TV Shows from IMDB")

            chrome_options = webdriver.ChromeOptions()
            chrome_driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

            # Load IMDb top TV page
            url = "https://www.imdb.com/chart/toptv/"
            chrome_driver.get(url)
            time.sleep(5)  # Wait for content to load

            # Find all list items with TV shows
            tv_shows = chrome_driver.find_elements(By.CSS_SELECTOR, "ul.ipc-metadata-list li.ipc-metadata-list-summary-item")

            results = []

            for li in tv_shows[:top_n]:  # Limit to top 100
                try:
                    img = li.find_element(By.TAG_NAME, "img")
                    alt = img.get_attribute("alt")  # e.g., "Bryan Cranston in Breaking Bad (2008)"
                    match = re.search(r'in (.+?) \(\d{4}\)', alt)
                    title = match.group(1) if match else alt

                    link = li.find_element(By.TAG_NAME, "a").get_attribute("href")
                    imdb_id = re.search(r'/title/(tt\d+)/', link).group(1)

                    results.append({
                        "title": title,
                        "imdb_id": imdb_id
                    })

                except Exception as e:
                    print(f"Skipped a row due to error: {e}")

            chrome_driver.quit()

            # Optional: Save to JSON
            with open(top_n_tv_shows_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
                log.info(f"successfully saved top {top_n} TV Shows from IMDB to Data/top_{top_n}_tv_shows.json")

            # load file after it was saved
            with open(top_n_tv_shows_path, "r", encoding="utf-8") as f:
                return json.load(f)

    def get_title(self, imdb_id: str, *, info: str | None = "main") -> Title:
        """
        Fetch a movie / series / episode by IMDb ID.

        Parameters
        ----------
        imdb_id : str
            Plain IMDb identifier, e.g. '0133093'.
        info : str | None
            What additional data to immediately pull.
            See Cinemagoer docs (update() keys).

        Returns
        -------
        Title
        """
        log.info("Fetching title %s …", imdb_id)
        try:
            data = self.ia.get_movie(imdb_id.replace("tt", ""))
            if info:
                self.ia.update(data, info)
            return Title(data)
        except IMDbError as exc:
            log.error("IMDb request failed: %s", exc)
            raise

    def search(self, query: str, limit: int = 10) -> List[Title]:
        """Search IMDb and return the first *limit* results as Title objects."""
        log.info("Searching IMDb for %r …", query)
        try:
            results = self.ia.search_movie(query)[:limit]
            return [Title(r) for r in results]
        except IMDbError as exc:
            log.error("Search failed: %s", exc)
            raise

    def get_series_episodes(self, series: Title | str, season: int | None = None, is_data_saved: bool = False) -> List | SERIES_NOT_FOUN_ERROR:
        if isinstance(series, Title):
            metadata_path = f"Data/{series.title}/Metadata/S{str(season)}_metadata.json"
        else:
            metadata_path = f"Data/{series}/Metadata/S{str(season)}_metadata.json"

        if os.path.exists(metadata_path):
            log.info(f"Load existing {metadata_path}")
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            log.info(f"Fetching {series} - Season {str(season)} episodes from IMDB")


        if isinstance(series, Title):
            url = f"http://www.omdbapi.com/?t={series.title.replace(' ', '+')}&Season={season}&apikey={OMDB_API_KEY}"
        else:
            url = f"http://www.omdbapi.com/?t={series.replace(' ', '+')}&Season={season}&apikey={OMDB_API_KEY}"

        response = requests.get(url)

        # Raise exception if request failed (optional but recommended)
        response.raise_for_status()

        # Parse JSON content
        json_data = response.json()

        if json_data.get('Error', 'N/A') == 'Series or season not found!':
            log.info(f"{series} Season {str(season)} - Series or season not found!")
            return SERIES_NOT_FOUN_ERROR

        # Extract episode info into a dict by episode number
        episodes_by_number = {
            int(ep["Episode"]): {
                "title": ep["Title"],
                "released": ep["Released"],
                "rating": ep["imdbRating"],
                "imdb_id": ep["imdbID"]
            }
            for ep in json_data["Episodes"]
        }

        episodes_imdb_data = []
        for episode_data in tqdm(episodes_by_number.values(), desc="Processing episodes"):
            # Fetch episode details using the IMDb API
            episodes_imdb_data.append(self.ia.get_episode(episode_data["imdb_id"].replace("tt", "")))

        if is_data_saved:
            # Construct the path
            if isinstance(series, Title):
                dir_path = f"Data/{series.title}/Metadata"
            else:
                dir_path = f"Data/{series}/Metadata"
            file_path = os.path.join(dir_path, f"S{str(season)}_metadata.json")

            # Create directories if they don't exist
            os.makedirs(dir_path, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(episodes_by_number, f, ensure_ascii=False, indent=4)
        return episodes_imdb_data

    def download_poster(self, title: Title | str, dest: Path | str) -> Path:
        """
        Download the primary poster of a title onto *dest*.

        Returns
        -------
        pathlib.Path of the downloaded file.
        """
        if isinstance(title, str):
            title = self.get_title(title)

        url = title.__dict__.get("full-size cover url")
        if not url:
            log.error("No poster URL available for this title.")
            return Path()

        dest = Path(dest).expanduser().with_suffix(".jpg")
        log.info("Downloading poster to %s …", dest)
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return dest


if __name__ == "__main__":
    client = CinemagoerClient()
    list_of_tv_shows_data = client.get_top_tv_shows(top_n=250)

    for tv_show_data in list_of_tv_shows_data:
        tv_show_name = client.get_title(imdb_id=tv_show_data["imdb_id"])
        for curr_season in range(1, 99):
            result = client.get_series_episodes(tv_show_name, season=curr_season, is_data_saved=True)
            if result == SERIES_NOT_FOUN_ERROR:
                break
