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
        logging.StreamHandler(),                       # print()‑like console output
        logging.FileHandler(_LOG_PATH, encoding="utf‑8")
    ],
)
log = logging.getLogger("CinemagoerClient")


# --------------------------------------------------------------------------- #
#  Domain Objects
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
#  Client
# --------------------------------------------------------------------------- #
class CinemagoerClient:
    """
    High‑level client providing *typed*, *logged* helpers around Cinemagoer.
    """

    def __init__(self, language: str | None = None) -> None:
        self.ia = Cinemagoer()
        if language:
            self.ia.set_locale(language)
            log.info("Locale set to %s", language)

    # ———————————————————————————— #
    #  Public API
    # ———————————————————————————— #
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

    def get_series_episodes(self, series: Title | str, season: int | None = None) -> List:
        url = f"http://www.omdbapi.com/?t={series.title.replace(' ', '+')}&Season={season}&apikey={OMDB_API_KEY}"
        response = requests.get(url)

        # Raise exception if request failed (optional but recommended)
        response.raise_for_status()

        # Parse JSON content
        json_data = response.json()

        # Extract episode info into a dict by episode number
        episodes_by_number = {
            int(ep["Episode"]): {
                "title": ep["Title"],
                "released": ep["Released"],
                "rating": float(ep["imdbRating"]),
                "imdb_id": ep["imdbID"]
            }
            for ep in json_data["Episodes"]
        }

        episodes_imdb_data = []
        for episode_data in tqdm(episodes_by_number.values(), desc="Processing episodes"):
            # Fetch episode details using the IMDb API
            episodes_imdb_data.append(self.ia.get_episode(episode_data["imdb_id"].replace("tt", "")))

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


# --------------------------------------------------------------------------- #
#  CLI demo (run `python cinemagoer_client.py`)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    client = CinemagoerClient()
    breaking_bed = client.get_title("tt0903747")  # Breaking Bad
    eps_s1 = client.get_series_episodes(breaking_bed, season=1)
    print(f"Breaking Bad – S1 has {len(eps_s1[1])} episode(s)")
