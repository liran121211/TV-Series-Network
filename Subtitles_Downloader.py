import json
import logging
import os
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import inflect
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# --------------------------------------------------------------------------- #
#  Environment Variables
# --------------------------------------------------------------------------- #
load_dotenv()  # loads from .env by default

# --------------------------------------------------------------------------- #
#  Logging configuration
# --------------------------------------------------------------------------- #
_LOG_PATH = Path(__file__).parent / "Logs" / (Path(__file__).stem + ".log")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(),  # print()‑like console output
        logging.FileHandler(_LOG_PATH, encoding="utf‑8")
    ],
)
logger = logging.getLogger("Logs/SubtitlesDownloader")

# --------------------------------------------------------------------------- #
#  Utils configuration
# --------------------------------------------------------------------------- #
# Create the inflect engine to convert numbers to words
p = inflect.engine()

# Create the dictionary
season_convert_table = {i: f"{p.number_to_words(p.ordinal(i))}-season" for i in range(1, 100)}

# get max cpu available for threading
MAX_CPU_THREADS = os.cpu_count()


class SubDownloader:
    def __init__(self):
        """
        Initialize the APIKeyManager.
        :param usage_limit_per_key: Max number of uses allowed per key.
        """

        self.available_api_keys = {
            os.getenv("SUBDL_API_KEY_1"): {'status': 'Available', 'usage': 0},
            os.getenv("SUBDL_API_KEY_2"): {'status': 'Available', 'usage': 0},
            os.getenv("SUBDL_API_KEY_3"): {'status': 'Available', 'usage': 0},
            os.getenv("SUBDL_API_KEY_4"): {'status': 'Available', 'usage': 0},
            os.getenv("SUBDL_API_KEY_5"): {'status': 'Available', 'usage': 0},
            os.getenv("SUBDL_API_KEY_6"): {'status': 'Available', 'usage': 0},
            os.getenv("SUBDL_API_KEY_7"): {'status': 'Available', 'usage': 0},
            os.getenv("SUBDL_API_KEY_8"): {'status': 'Available', 'usage': 0},
        }

        # utilize usage
        self.set_api_keys_usage()
        self.active_api_key = self.get_active_api_key()

    def get_active_api_key(self):
        for api_key, api_key_data in self.available_api_keys.items():
            if api_key_data['status'] == 'Available':
                logger.warning("Key is exhausted, setting different api key...")
                return api_key
        raise LookupError('No available api keys for use. try tomorrow or add more.')

    def set_api_keys_usage(self):
        for api_key, api_key_data in self.available_api_keys.items():
            self.check_usage_limit(api_key=api_key)

    def check_usage_limit(self, api_key):
        status_url = f'https://api.subdl.com/api/v1/subtitles?api_key={api_key}'

        # Make the GET request to the SubDL API
        response = requests.get(status_url)

        # Parse the JSON response
        if response.status_code == 200:
            result = response.json()

            if result.get('message', '') == "Daily Limit" or result.get('statusCode', '') == "429":
                self.available_api_keys[api_key] = {'status': 'Exhausted', 'usage': 1000}
                return 'Exhausted'
            else:
                return 'Available'

        return 'ERROR'

    def is_subtitle_exists(self, imdb_id):
        BASE_URL = 'https://api.subdl.com/api/v1/subtitles'

        params = {
            "api_key": self.active_api_key,
            "imdb_id": imdb_id,
            "languages": 'EN',  # separate them by comma
        }

        # Make the GET request to the SubDL API
        response = requests.get(BASE_URL, params=params)

        # Parse the JSON response
        if response.status_code == 200:
            result = response.json()

            if result.get('message', '') == "Daily Limit" or result.get('statusCode', '') == "429":
                self.set_api_keys_usage()
                self.active_api_key = self.get_active_api_key()
                self.is_subtitle_exists(imdb_id=imdb_id)

            print(f'Checking Title: {imdb_id}...')
            if result.get('error', 'N/A') == "can't find movie or tv":
                return False

            if result.get('subtitles', 'N/A') != 'N/A':
                if len(result['subtitles']) > 0:
                    return True
        return False


    def get_subtitles(self, film_name=None, file_name=None, sd_id=None, imdb_id=None, tmdb_id=None, season_number=None, episode_number=None, type=None, year=None, languages=None, subs_per_page=10):
        # Define the base URL for the SubDL API
        BASE_URL = "https://api.subdl.com/api/v1/subtitles"

        # Define the prefix for subtitle download links
        LINK_PREFIX = "https://dl.subdl.com"

        # Construct the query parameters based on provided arguments
        params = {
            "api_key": self.active_api_key,
            "film_name": film_name,
            "file_name": file_name,
            "sd_id": sd_id,
            "imdb_id": imdb_id,
            "tmdb_id": tmdb_id,
            "season_number": season_number,
            "episode_number": episode_number,
            "type": type,  # movie or tv
            "year": year,
            "languages": languages,  # seperate them by comma
            "subs_per_page": min(subs_per_page, 30)  # Limit subs_per_page to maximum 30
        }

        # Make the GET request to the SubDL API
        response = requests.get(BASE_URL, params=params)

        # Parse the JSON response
        if response.status_code == 200:
            result = response.json()

            if result.get('message', '') == "Daily Limit" or result.get('statusCode', '') == "429":
                self.set_api_keys_usage()
                self.active_api_key = self.get_active_api_key()
                self.is_subtitle_exists(imdb_id=imdb_id)

            if result["status"]:
                for subtitle in result["subtitles"]:
                    if "url" in subtitle:
                        return subtitle['name'], LINK_PREFIX + subtitle["url"]
        else:
            return 'N/A', 'N/A'


def get_subtitles_of_tv_show(imdb_id, season_id: int, episode_id: int, language: str):
    search_url = f"https://subdl.com/search/{imdb_id}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    }

    cookies = {
        "_ga_N02LL12MHK": "GS2.1.s1748318973$o2$g1$t1748318980$j0$l0$h0",
        "_ga": "GA1.2.259489255.1748288010",
        "_gid": "GA1.2.856341465.1748288011",
        "_gat_gtag_UA_57662958_1": "1",
    }

    try:
        # Step 1: Get search results page
        response = requests.get(search_url, headers=headers, cookies=cookies)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        search_page_first_result_html_token, search_page_first_result_html_type = "container mx-auto px-5 flex flex-col gap-1 rounded-md", 'div'

        # Step 2: Find the first result link
        first_result = \
        soup.select_one(_class=search_page_first_result_html_token, selector=search_page_first_result_html_type).select(
            selector='a')[0].attrs.get('href', 'N/A')
        if first_result == 'N/A':
            logger.critical("No results found for the given IMDb ID")
            return 'RNE', 'RNE'  # Result Not Exist

        title_page_link = "https://subdl.com" + first_result
        response = requests.get(title_page_link, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the <div> with the specified class
        target_div = soup.find('div', class_='flex flex-col gap-4 flex-grow mt-0 md:-mt-[40px]')

        # Get all <a> tags inside it
        if target_div:
            links = target_div.find_all('a')
            hrefs = [a.get('href') for a in links if a.get('href')]
        else:
            hrefs = []

        if season_id > len(hrefs) or season_id < 1:
            return 'SNE', 'SNE'  # Season Not Exist

        season_page_link = title_page_link + '/' + season_convert_table.get(season_id, 'N/A') + '/' + language
        response = requests.get(season_page_link, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        if soup.select_one(selector='h2', _class='text-2xl font-bold'):
            if soup.select_one(selector='h2', _class='text-2xl font-bold').text.strip() == 'No subtitles found':
                logger.critical("No subtitles found for the given season and language")
                return 'NSF', 'NSF'  # No Show Found

        subtitles_list_li_element = soup.find_all("li",
                                                  class_="flex justify-between flex-col lg:flex-row gap-1 items-start lg:items-center")
        for selected_subtitles in subtitles_list_li_element:
            selected_subtitles_link = selected_subtitles.select_one("a.inline-flex")["href"]
            selected_subtitles_name = selected_subtitles.select_one("h4", class_="inline-flex").text.strip()

            if 1 <= episode_id <= 9:
                if 'E0' + str(episode_id) in selected_subtitles_name:
                    selected_subtitles_name = selected_subtitles_name.replace(' ', '_') + 'S' + str(
                        season_id) + 'E' + str(episode_id) + '.zip'
                    return selected_subtitles_name, selected_subtitles_link
            else:
                if 'E' + str(episode_id) in selected_subtitles_name:
                    selected_subtitles_name = selected_subtitles_name.replace(' ', '_') + 'S' + str(
                        season_id) + 'E' + str(episode_id) + '.zip'
                    return selected_subtitles_name, selected_subtitles_link
                else:
                    return 'NTF', 'NTF'  # Non-Template Filename

        return 'N/A', 'N/A'

    except requests.RequestException as e:
        logger.exception(f"Request error: {e}")
    except Exception as ex:
        logger.exception(f"Error: {ex}")
    return None


def update_season_metadata_with_subtitles(base_path='Data'):
    zip_pattern = re.compile(r'S(\d{1,2})E(\d{1,2})', re.IGNORECASE)

    for tv_show_folder in os.listdir(base_path):
        show_path = os.path.join(base_path, tv_show_folder)
        subtitles_base = os.path.join(show_path, 'Subtitles')
        metadata_base = os.path.join(show_path, 'Metadata')

        if not os.path.isdir(subtitles_base) or not os.path.isdir(metadata_base):
            continue

        # Traverse Subtitles/SX folders
        for season_folder in os.listdir(subtitles_base):
            season_path = os.path.join(subtitles_base, season_folder)
            if not os.path.isdir(season_path):
                continue

            for file in os.listdir(season_path):
                if not file.endswith('.zip'):
                    continue

                match = zip_pattern.search(file)
                if not match:
                    continue

                season_num = int(match.group(1))
                if season_num == 0:
                    season_num = 1  # Adjust season number if needed

                episode_num = int(match.group(2))
                if episode_num == 0:
                    episode_num = 1  # Adjust episode number if needed

                metadata_file_name = f'S{season_num}_metadata.json'
                metadata_file_path = os.path.join(metadata_base, metadata_file_name)

                if not os.path.isfile(metadata_file_path):
                    print(f"Metadata file not found: {metadata_file_path}")
                    continue

                try:
                    with open(metadata_file_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    episode_key = str(episode_num)
                    if episode_key in metadata:
                        print(f"Updating metadata for {metadata_file_path} -> Episode {episode_key}")
                        metadata[episode_key]['subtitles_exists'] = True
                        metadata[episode_key]['subtitles_full_path'] = os.path.abspath(os.path.join(season_path, file))

                        # Save updated metadata
                        with open(metadata_file_path, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=4, ensure_ascii=False)

                except Exception as e:
                    print(f"Failed to update {metadata_file_path}: {e}")


def is_tv_show_folder_exists(base_path, substring):
    for entry in os.listdir(base_path):
        full_path = os.path.join(base_path, entry)
        if os.path.isdir(full_path) and substring in entry:
            return True
    return False


def download_with_selenium(info_page_url, download_dir):
    # Set Chrome preferences for automatic download
    chrome_options = Options()
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": os.path.abspath(download_dir),
        "download.prompt_for_download": False,
        "directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    # Optional: run headless (no browser window)
    # chrome_options.add_argument("--headless")

    # Start Chrome
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(info_page_url)

    # Wait for the download to finish (increase time for large files)
    time.sleep(10)  # or use a more robust wait
    logger.info(f"✅ ZIP saved to: {download_dir}")

    driver.quit()


def download_zip_to_folder(url, save_to_folder, zip_name):
    # Ensure destination folder exists
    os.makedirs(save_to_folder, exist_ok=True)

    zip_name = re.sub(r'[<>:"|?*]', '_', str(zip_name))  # Sanitize filename
    zip_name = re.sub(r'[<>:"/\\|?*]', '_', zip_name)  # Replace spaces with underscores

    # Determine filename from URL or headers
    local_filename = url.split("/")[-1]
    save_path = os.path.join(save_to_folder, zip_name)

    if url == 'N/A':
        logger.critical(f'invalid url: {url}')
        return save_path

    if os.path.exists(save_path):
        logger.info(f"ZIP already exists in: {save_path}")
        return save_path

    # Download the file
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": 'https://google.com'
    }

    cookies = {
        "PHPSESSID": "0a2bnUhJwukK6ss-3r50eddFV74",
        "download-counter": "1",
        "server": "1",
    }

    session = requests.Session()
    session.headers.update(headers)
    session.cookies.update(cookies)

    with session.get(url, stream=True) as r:
        r.raise_for_status()
        if r.status_code != 200:
            logger.exception(f"Error downloading subtitles.")
            return 'SPNE'  # Save path not exists

        if 'zip' not in r.headers.get('Content-Type', '') and not local_filename.endswith(".zip"):
            logger.critical("Downloaded file is not a ZIP")
            download_with_selenium(info_page_url=url, download_dir=save_to_folder)
            return save_path

        try:
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            logger.exception(f"Error writing to file {save_path}: {e}")
            return 'SPNE'  # Save path not exists

    logger.info(f"✅ ZIP saved to: {save_path}")
    return save_path


def fetch_list_season_metadata_files(folder_path):
    if not os.path.exists(folder_path):
        logger.error(f"Folder does not exist: {folder_path}")
        return [], 0

    pattern = re.compile(r'^S\d{1,2}_metadata\.json$')
    seasons_list = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and pattern.match(f)
    ]
    return seasons_list, len(seasons_list)


def process_episode(args):
    tv_show_name, tv_show_id, season_id, episode_id, season_metadata_path, base_path = args

    try:
        # Load the JSON for this season (ideally, do this once per season, but for simplicity, per episode)
        with open(season_metadata_path, "r", encoding="utf-8") as f:
            json_season_data = json.load(f)

        # Check if subtitles already exist
        try:
            if json_season_data[str(episode_id)]['subtitles_exists'] is True and json_season_data[str(episode_id)]['subtitles_full_path'] != '':
                return f"Subtitles already exist for {tv_show_name} S{season_id} E{episode_id}, skipping..."
        except KeyError:
            pass

        # Fetch subtitle info
        # st_name, st_link = get_subtitles_of_tv_show(imdb_id=tv_show_id, season_id=season_id, episode_id=episode_id, language='english')
        st_name, st_link = sub_dl.get_subtitles(imdb_id=tv_show_id, season_number=season_id, episode_number=episode_id, languages='EN')
        time.sleep(5)

        # Handle possible responses (NSF, NTF, SNE, RNE, etc.)
        if st_name in ['N/A', 'NSF', 'NTF', 'SNE', 'RNE'] or st_link in ['N/A', 'NSF', 'NTF', 'SNE', 'RNE']:
            return f"Skipping {tv_show_name} S{season_id} E{episode_id} due to status {st_name or st_link}"

        # Prepare directory for subtitles
        current_subtitles_path = os.path.normpath(
            os.path.join(os.getcwd(), base_path, tv_show_name, 'Subtitles', 'S' + str(season_id)))
        os.makedirs(current_subtitles_path, exist_ok=True)

        # Download subtitles
        download_zip_to_folder(url=st_link, save_to_folder=current_subtitles_path, zip_name=st_name)

        # Update JSON metadata
        json_season_data[str(episode_id)]['subtitles_exists'] = True
        json_season_data[str(episode_id)]['subtitles_full_path'] = os.path.abspath(
            os.path.join(current_subtitles_path, st_name))

        # Write back the season metadata (make sure only one thread writes at a time!)
        with open(season_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(json_season_data, f, indent=4, ensure_ascii=False)

        return f"Downloaded subtitles for {tv_show_name} S{season_id} E{episode_id}"

    except Exception as e:
        return f"Error processing {tv_show_name} S{season_id} E{episode_id}: {e}"


# Example usage
if __name__ == "__main__":
    sub_dl = SubDownloader()
    # season_pattern = re.compile(r'^S\d{1,2}_metadata\.json$')
    # base_path = r'C:\Users\mor21\PycharmProjects\BigData_TV_Series_Project\Data'
    # work_list = []
    #
    # with open('Utils/approve_relevant_tv_shows.json', "r", encoding="utf-8") as tv_show_f:
    #     json_tv_shows_data = json.load(tv_show_f)
    #     for tv_show in json_tv_shows_data:
    #         tv_show_name = re.sub(r'[\\/:*?"<>|]', '_', tv_show['title'])
    #         tv_show_id = tv_show["imdb_id"]
    #
    #         if is_tv_show_folder_exists(base_path='Data', substring=tv_show_name):
    #             full_tv_show_folder_path = os.path.join('Data', tv_show_name, 'Metadata')
    #             seasons_files, max_seasons = fetch_list_season_metadata_files(full_tv_show_folder_path)
    #             for season_id, season_metadata_path in enumerate(seasons_files, start=1):
    #                 if os.path.isfile(season_metadata_path):
    #                     with open(season_metadata_path, "r", encoding="utf-8") as f:
    #                         json_season_data = json.load(f)
    #                         first_json_key = next(iter(json_season_data.items()))[0]
    #                         for episode_id, episode_data in enumerate(json_season_data.items(), start=int(first_json_key)):
    #                             # If subtitles need to be fetched, add to work list
    #                             try:
    #                                 if json_season_data[str(episode_id)]['subtitles_exists'] is True and json_season_data[str(episode_id)]['subtitles_full_path'] != '':
    #                                     continue
    #                             except KeyError as e:
    #                                 continue
    #                             work_list.append((tv_show_name, tv_show_id, season_id, episode_id, season_metadata_path, base_path))
    #
    # # Now, process in parallel!
    # with ThreadPoolExecutor(max_workers=MAX_CPU_THREADS) as executor:
    #     future_to_work = {executor.submit(process_episode, args): args for args in work_list}
    #     for future in as_completed(future_to_work):
    #         print(future.result())
