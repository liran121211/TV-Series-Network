import json
import os
import re
import shutil
import zipfile

import requests
from bs4 import BeautifulSoup


def get_subtitles_of_tv_show(imdb_id, season_id: int, episode_id: int, language: str):
    search_url = f"https://subdl.com/search/{imdb_id}"
    headers = {
        "User-Agent": "Mozilla/5.0"
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
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        search_page_first_result_html_token, search_page_first_result_html_type = "container mx-auto px-5 flex flex-col gap-1 rounded-md", 'div'

        # Step 2: Find the first result link
        first_result = soup.select_one(_class=search_page_first_result_html_token, selector=search_page_first_result_html_type).select(selector='a')[0].attrs.get('href', 'N/A')
        if first_result == 'N/A':
            raise ValueError("No results found for the given IMDb ID")

        title_page_link = "https://subdl.com" + first_result
        response = requests.get(title_page_link, headers=headers)
        response.raise_for_status()
        # soup = BeautifulSoup(response.text, "html.parser")

        season_page_link = title_page_link + '/' + season_convert_table.get(season_id, 'N/A') + '/' + language
        response = requests.get(season_page_link, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        subtitles_list_li_element = soup.findAll("li", class_="flex justify-between flex-col lg:flex-row gap-1 items-start lg:items-center")
        for selected_subtitles in subtitles_list_li_element:
            selected_subtitles_link = selected_subtitles.select_one("a.inline-flex")["href"]
            selected_subtitles_name = selected_subtitles.select_one("h4", class_="inline-flex").text.strip()

            if 1 <= episode_id <= 9:
                if 'E0' + str(episode_id) in selected_subtitles_name:
                    return selected_subtitles_name, selected_subtitles_link
            else:
                if 'E' + str(episode_id) in selected_subtitles_name:
                    return selected_subtitles_name, selected_subtitles_link

        return 'N/A', 'N/A'

    except requests.RequestException as e:
        print(f"Request error: {e}")
    except Exception as ex:
        print(f"Error: {ex}")
    return None


def is_tv_show_folder_exists(base_path, substring):
    for entry in os.listdir(base_path):
        full_path = os.path.join(base_path, entry)
        if os.path.isdir(full_path) and substring in entry:
            return True
    return False


def download_zip_to_folder(url, save_to_folder, zip_name):
    # Ensure destination folder exists
    os.makedirs(save_to_folder, exist_ok=True)

    # Determine filename from URL or headers
    local_filename = url.split("/")[-1]
    save_path = os.path.join(save_to_folder, zip_name)

    # Download the file
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        if 'zip' not in r.headers.get('Content-Type', '') and not local_filename.endswith(".zip"):
            raise ValueError("Downloaded file is not a ZIP")

        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"✅ ZIP saved to: {save_path}")
    return save_path

# Example usage
if __name__ == "__main__":
    with open('Data/top_250_tv_shows.json', "r", encoding="utf-8") as f:
        json_tv_shows_data = json.load(f)
        for tv_show in json_tv_shows_data:
            tv_show_name = re.sub(r"\s*\(\d{4}\)$", "", tv_show["title"])
            tv_show_id = tv_show["imdb_id"]

            if is_tv_show_folder_exists(base_path='Data', substring=tv_show_name):
                full_tv_show_folder_path = os.path.join('Data', tv_show_name, 'Metadata')

                season_id = 0
                for season_metadata in os.listdir(full_tv_show_folder_path):
                    season_metadata_path = os.path.join(full_tv_show_folder_path, season_metadata)

                    if os.path.isfile(season_metadata_path):
                        with open(season_metadata_path, "r", encoding="utf-8") as f:
                            json_season_data = json.load(f)
                            season_id += 1

                            for episode_id, episode_data in enumerate(json_season_data, 1):
                                st_name, st_link = get_subtitles_of_tv_show(imdb_id=tv_show_id, season_id=season_id, episode_id=episode_id, language='english')

                                if st_name and st_name:
                                    # Create directories if they don't exist
                                    current_subtiitles_path = os.path.join(os.getcwd(), full_tv_show_folder_path, '../Subtitles', 'S' + str(season_id))
                                    os.makedirs(current_subtiitles_path, exist_ok=True)

                                    zip_name = st_name.replace(' ', '_') + 'S' + str(season_id) + 'E' + str(episode_id) + '.zip'
                                    download_zip_to_folder(url=st_link, save_to_folder=current_subtiitles_path, zip_name=zip_name)


