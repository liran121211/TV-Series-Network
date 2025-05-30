import json
import logging
import os
import re
import shutil
import zipfile
from pathlib import Path

import requests
from bs4 import BeautifulSoup

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
logger = logging.getLogger("Logs/SubtitlesDownloader")

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

    season_convert_table = {
        1: 'first-season',
        2: 'second-season',
        3: 'third-season',
        4: 'fourth-season',
        5: 'fifth-season',
        6: 'sixth-season',
        7: 'seventh-season',
        8: 'eighth-season',
        9: 'ninth-season',
        10: 'tenth-season',
    }

    try:
        # Step 1: Get search results page
        response = requests.get(search_url, headers=headers, cookies=cookies)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        search_page_first_result_html_token, search_page_first_result_html_type = "container mx-auto px-5 flex flex-col gap-1 rounded-md", 'div'

        # Step 2: Find the first result link
        first_result = soup.select_one(_class=search_page_first_result_html_token, selector=search_page_first_result_html_type).select(selector='a')[0].attrs.get('href', 'N/A')
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
                return 'NSF', 'NSF' # No Show Found

        subtitles_list_li_element = soup.find_all("li", class_="flex justify-between flex-col lg:flex-row gap-1 items-start lg:items-center")
        for selected_subtitles in subtitles_list_li_element:
            selected_subtitles_link = selected_subtitles.select_one("a.inline-flex")["href"]
            selected_subtitles_name = selected_subtitles.select_one("h4", class_="inline-flex").text.strip()

            if 1 <= episode_id <= 9:
                if 'E0' + str(episode_id) in selected_subtitles_name:
                    return selected_subtitles_name, selected_subtitles_link
            else:
                if 'E' + str(episode_id) in selected_subtitles_name:
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
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        if 'zip' not in r.headers.get('Content-Type', '') and not local_filename.endswith(".zip"):
            logger.critical("Downloaded file is not a ZIP")
            raise ValueError("Downloaded file is not a ZIP")

        try:
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            logger.exception(f"Error writing to file {save_path}: {e}")
            return

    logger.info(f"✅ ZIP saved to: {save_path}")
    return save_path


def fetch_list_season_metadata_files(folder_path):
    pattern = re.compile(r'^S\d{1,2}_metadata\.json$')
    seasons_list = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and pattern.match(f)
    ]
    return seasons_list, len(seasons_list)


# Example usage
if __name__ == "__main__":
    # update_season_metadata_with_subtitles(base_path=r'C:\Users\mor21\PycharmProjects\BigData_TV_Series_Project\Data')

    season_pattern = re.compile(r'^S\d{1,2}_metadata\.json$')
    flag = False
    with open('Data/top_250_tv_shows.json', "r", encoding="utf-8") as tv_show_f:
        json_tv_shows_data = json.load(tv_show_f)
        for tv_show in json_tv_shows_data:
            tv_show_name = re.sub(r"\s*\(\d{4}\)$", "", tv_show["title"])
            tv_show_id = tv_show["imdb_id"]

            # if not 'Ted Lasso' in tv_show_name:
            #     continue
            if 'Ted Lasso' not in tv_show_name and not flag:
                continue

            if not flag:
                flag = True

                logger.info(f"Skipping {tv_show_name} as it is in the ignore list")
                continue

            if is_tv_show_folder_exists(base_path='Data', substring=tv_show_name):
                full_tv_show_folder_path = os.path.join('Data', tv_show_name, 'Metadata')

                seasons_files, max_seasons = fetch_list_season_metadata_files(full_tv_show_folder_path)
                for season_id, season_metadata_path in enumerate(seasons_files, start=1):

                    if os.path.isfile(season_metadata_path):
                        with open(season_metadata_path, "r", encoding="utf-8") as f:
                            json_season_data = json.load(f)

                            for episode_id, episode_data in enumerate(json_season_data, start=0):
                                try:
                                    json_season_data[str(episode_id)]['subtitles_exists']
                                except KeyError:
                                    episode_id += 1

                                try:
                                    if json_season_data[str(episode_id)]['subtitles_exists'] is True and json_season_data[str(episode_id)]['subtitles_full_path'] != '':
                                        logger.info(f"Subtitles already exist for {tv_show_name} S{season_id}, skipping...")
                                        continue
                                except KeyError:
                                    continue

                                st_name, st_link = get_subtitles_of_tv_show(imdb_id=tv_show_id, season_id=season_id, episode_id=episode_id, language='english')
                                if st_name == 'N/A' or st_link == 'N/A':
                                    logger.critical(f"Failed to fetch subtitles for {tv_show_name} S{season_id} E{episode_id}")
                                    continue

                                if st_name == 'NSF' or st_link == 'NSF':
                                    break

                                if st_name == 'NTF' or st_link == 'NTF':
                                    logger.info(f"Non-Template Filename for {tv_show_name} S{season_id} E{episode_id}")
                                    continue

                                if st_name == 'SNE' or st_link == 'SNE':
                                    logger.info(f"Season {season_id} does not exist for the given TV show")
                                    break

                                if st_name == 'RNE' or st_link == 'RNE':
                                    logger.info(f"Search result for Season {season_id} does not exist for the given TV show")
                                    break

                                if st_name and st_name:
                                    # Create directories if they don't exist
                                    current_subtiitles_path = os.path.normpath(os.path.join(os.getcwd(), full_tv_show_folder_path, '../Subtitles', 'S' + str(season_id)))
                                    os.makedirs(current_subtiitles_path, exist_ok=True)

                                    zip_name = st_name.replace(' ', '_') + 'S' + str(season_id) + 'E' + str(episode_id) + '.zip'
                                    download_zip_to_folder(url=st_link, save_to_folder=current_subtiitles_path, zip_name=zip_name)

                                    json_season_data[str(episode_id)]['subtitles_exists'] = True
                                    json_season_data[str(episode_id)]['subtitles_full_path'] = os.path.abspath(os.path.join(current_subtiitles_path, zip_name))
                                    # Save updated metadata
                                    with open(season_metadata_path, 'w', encoding='utf-8') as f:
                                        json.dump(json_season_data, f, indent=4, ensure_ascii=False)


