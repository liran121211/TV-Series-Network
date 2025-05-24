from __future__ import annotations

import glob
import os
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from collections import Counter
from scipy.stats import entropy
import requests
from imdb import Cinemagoer, IMDbError, IMDbDataAccessError
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
CURR_OMDB_API_KEY_IDX = OMDB_API_KEY.split('|')[1]
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


def sanitize_tv_show_name(name) -> str:
    # Replace invalid characters with underscore or remove them
    return re.sub(r'[\\/:*?"<>|]', '_', name)


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
        self.logger = logging.getLogger("Logs/IMDB_Analyzer")
        if language:
            self.ia.set_locale(language)
            self.logger.info("Locale set to %s", language)

    def get_top_tv_shows(self, top_n: int = 100):
        top_n_tv_shows_path = f"Data/top_{top_n}_tv_shows.json"

        if os.path.exists(top_n_tv_shows_path):
            self.logger.info(f"Load existing {top_n_tv_shows_path}")
            with open(top_n_tv_shows_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            self.logger.info(f"Fetching top {top_n} TV Shows from IMDB")

            chrome_options = webdriver.ChromeOptions()
            chrome_driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

            # Load IMDb top TV page
            url = "https://www.imdb.com/chart/toptv/"
            chrome_driver.get(url)
            time.sleep(5)  # Wait for content to load

            # Find all list items with TV shows
            tv_shows = chrome_driver.find_elements(By.CSS_SELECTOR,
                                                   "ul.ipc-metadata-list li.ipc-metadata-list-summary-item")

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
                self.logger.info(f"successfully saved top {top_n} TV Shows from IMDB to Data/top_{top_n}_tv_shows.json")

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
        self.logger.info("Fetching title %s …", imdb_id)
        try:
            data = self.ia.get_movie(imdb_id.replace("tt", ""))
            if info:
                self.ia.update(data, info)
            return Title(data)
        except IMDbError as exc:
            self.logger.error("IMDb request failed: %s", exc)
            raise

    def get_episode_metadata(self, metadata_file_path: str):
        """
        Fetches episode metadata from IMDb for all entries in a JSON metadata file.

        If episode metadata file already exists, it loads it instead of fetching.
        Saves each episode's metadata to a separate file and returns the last metadata entry.
        """
        with open(metadata_file_path, "r", encoding="utf-8") as f:
            series_data = json.load(f)

        for idx, (key, metadata) in enumerate(series_data.items(), start=1):
            imdb_id = metadata.get("imdb_id", "").replace("tt", "")
            title = metadata.get("title", f"Episode {idx}")

            # Construct output path
            output_path = metadata_file_path.replace("_metadata", f"_E{idx}_metadata")

            # If metadata file already exists, load it
            if os.path.exists(output_path):
                self.logger.info("Using cached metadata for: %s", title)
            else:
                episode_metadata = {
                    'cast': dict(),
                    'geners': list(),
                    'runtimes': 0,
                    'number of episodes': 0,
                    'plot': 'N/A',
                    'genres': 'N/A',
                    'imdbID': 'tt0000000',
                    'rating': 0,
                    'votes': 0,

                }
                self.logger.info("Fetching metadata for: %s", title)
                episode_imdb_data = self.ia.get_movie(imdb_id)
                with open(output_path, "w", encoding="utf-8") as out_file:
                    for person in episode_imdb_data.data["cast"]:
                        episode_metadata["cast"][person.personID] = person['name']

                    episode_metadata['geners'] = episode_imdb_data['genres']
                    episode_metadata['runtimes'] = episode_imdb_data['runtimes']
                    episode_metadata['number_of_episodes'] = episode_imdb_data['number of episodes']
                    episode_metadata['plot'] = episode_imdb_data['plot']
                    episode_metadata['geners'] = episode_imdb_data['genres']
                    episode_metadata['imdbID'] = episode_imdb_data['imdbID']
                    episode_metadata['rating'] = episode_imdb_data['rating']
                    episode_metadata['votes'] = episode_imdb_data['votes']

                    # save the data
                    self.logger.info(f"Extracted Metadata of {output_path}")
                    json.dump(episode_metadata, out_file)

    def search(self, query: str, limit: int = 10) -> List[Title]:
        """Search IMDb and return the first *limit* results as Title objects."""
        self.logger.info("Searching IMDb for %r …", query)
        try:
            results = self.ia.search_movie(query)[:limit]
            return [Title(r) for r in results]
        except IMDbError as exc:
            self.logger.error("Search failed: %s", exc)
            raise

    def get_series_episodes(self, series: Title | str, season: int | None = None, is_data_saved: bool = False) -> List | SERIES_NOT_FOUN_ERROR:
        if isinstance(series, Title):
            metadata_path = f"Data/{series.title}/Metadata/S{str(season)}_metadata.json"
            series.title = sanitize_tv_show_name(series.title)
        else:
            metadata_path = f"Data/{series}/Metadata/S{str(season)}_metadata.json"
            series = sanitize_tv_show_name(series)

        if os.path.exists(metadata_path):
            self.logger.info(f"Load existing {metadata_path}")
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            self.logger.info(f"Fetching {series} - Season {str(season)} episodes from IMDB")

        if isinstance(series, Title):
            url = f"http://www.omdbapi.com/?t={series.title.replace(' ', '+')}&Season={season}&apikey={CURR_OMDB_API_KEY_IDX}"
        else:
            url = f"http://www.omdbapi.com/?t={series.replace(' ', '+')}&Season={season}&apikey={CURR_OMDB_API_KEY_IDX}"

        response = requests.get(url)

        # Raise exception if request failed (optional but recommended)
        response.raise_for_status()

        # Parse JSON content
        json_data = response.json()

        if json_data.get('Error', 'N/A') == 'Series or season not found!':
            self.logger.info(f"{series} Season {str(season)} - Series or season not found!")
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
            try:
                episodes_imdb_data.append(self.ia.get_episode(episode_data["imdb_id"].replace("tt", "")))
            except IMDbDataAccessError:
                self.logger.error(f"IMDB URL is not available for this title {episode_data['imdb_id']}")
                continue

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
            self.logger.error("No poster URL available for this title.")
            return Path()

        dest = Path(dest).expanduser().with_suffix(".jpg")
        self.logger.info("Downloading poster to %s …", dest)
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return dest

    def extract_imdb_features(self, episode: Dict[str, Any]) -> pd.Series:
        self.logger.info('Extracting features of an episode')

        # Extract Cast Features
        def feature_cast_size(cast: Dict[str, str]) -> int:
            return len(cast)

        def feature_cast_unique_names_count(cast: Dict[str, str]) -> int:
            return len(set(cast.values()))

        def feature_cast_id_entropy(cast: Dict[str, str]) -> float:
            ids = list(cast.keys())
            counts = Counter(ids)
            probabilities = np.array(list(counts.values())) / len(ids)
            return float(entropy(probabilities, base=2)) if len(ids) > 1 else 0.0

        def feature_cast_character_name_length_avg(cast: Dict[str, str]) -> float:
            lengths = [len(name) for name in cast.values()]
            return float(np.mean(lengths)) if lengths else 0.0

        def feature_cast_character_name_token_count_avg(cast: Dict[str, str]) -> float:
            token_counts = [len(name.split()) for name in cast.values()]
            return float(np.mean(token_counts)) if token_counts else 0.0

        # Extract Genre Features
        def feature_genre_count(genres: List[str]) -> int:
            return len(genres)

        def feature_is_multigenre(genres: List[str]) -> int:
            return int(len(genres) > 1)

        # Extract Technical Features
        def feature_runtime_minutes(runtimes: list) -> int:
            try:
                return int(runtimes[0]) if runtimes else 0
            except (ValueError, IndexError):
                return 0

        def feature_number_of_episodes(episode: Dict[str, Any]) -> int:
            return int(episode.get("number_of_episodes", 0))

        _cast = episode.get("cast", {})
        _genres = episode.get("genres", [])

        return pd.Series({
            "cast_size": feature_cast_size(_cast),
            "cast_unique_names_count": feature_cast_unique_names_count(_cast),
            "cast_id_entropy": feature_cast_id_entropy(_cast),
            "cast_character_name_length_avg": feature_cast_character_name_length_avg(_cast),
            "cast_character_name_token_count_avg": feature_cast_character_name_token_count_avg(_cast),
            "genre_count": feature_genre_count(_genres),
            "is_multigenre": feature_is_multigenre(_genres),
            "runtime_minutes": feature_runtime_minutes(episode.get("runtimes", [])),
            "number_of_episodes": feature_number_of_episodes(episode),
        })


if __name__ == "__main__":
    client = CinemagoerClient()

    # extract metadata of each TV Show
    # list_of_tv_shows_data = client.get_top_tv_shows(top_n=250)
    # for tv_show_data in list_of_tv_shows_data:
    #     tv_show_name = client.get_title(imdb_id=tv_show_data["imdb_id"])
    #     for curr_season in range(1, 250):
    #         result = client.get_series_episodes(tv_show_name, season=curr_season, is_data_saved=True)
    #         if result == SERIES_NOT_FOUN_ERROR:
    #             break

    # extract metadata of each episode of TV Show
    matched_files = []
    for root, dirs, files in os.walk(r'C:\Users\mor21\PycharmProjects\BigData_TV_Series_Project\Data'):
        if os.path.basename(root) == "Metadata":
            for file in files:
                if re.match(r"S\d{1}_metadata\.json$", file):
                    matched_files.append(os.path.join(root, file))

    for json_metadata_path in matched_files:
        try:
            client.get_episode_metadata(metadata_file_path=json_metadata_path)
        except Exception:
            x = 1

    # with open(
    #         r"C:\Users\mor21\PycharmProjects\BigData_TV_Series_Project\Data\Avatar_ The Last Airbender\Metadata\S2_E1_metadata.json",
    #         "r", encoding="utf-8") as f:
    #     x = client.extract_imdb_features(json.load(f))

