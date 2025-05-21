import requests
from bs4 import BeautifulSoup


def get_subdl_first_result_page(imdb_id, season_id: int, episode_id: int, language: str):
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

        season_page_link = title_page_link + '/' + season_convert_table[season_id] + '/' + language
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

# Example usage
if __name__ == "__main__":
    st_name, st_link = get_subdl_first_result_page(imdb_id='tt0098904', season_id=2, episode_id=2, language='english')
    if st_name and st_name:
        print("Successfully retrieved the subtitle page.")
        print("Subtitles Name: ", st_name)
        print("Subtitles Link: ", st_link)

