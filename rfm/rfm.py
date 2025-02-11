from bs4 import BeautifulSoup
import random
import requests


def load_soup(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    return BeautifulSoup(response.text, "html.parser")


def get_rndm_mixkit_rfmp3_link():
    query_url = "https://mixkit.co/free-stock-music/"
    soup = load_soup(query_url)
    div_list = soup.find_all("div", {"data-controller": "audio-player"})
    mp3_links_mixkit = []
    for link in div_list:
        mp3_links_mixkit.append(link["data-audio-player-preview-url-value"])
    return random.choice(mp3_links_mixkit)


def download_rfm_mp3(url):
    response = requests.get(url)
    response.raise_for_status()
    name = url.split("/")[-1]
    with open("rfm/music/" + name, "wb") as f:
        f.write(response.content)


if __name__ == "__main__":
    url = get_rndm_mixkit_rfmp3_link()
    download_rfm_mp3(url)
    print("Downloaded RFM mp3")
