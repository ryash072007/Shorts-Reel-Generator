import os
from bs4 import BeautifulSoup
import random
import requests

# Complete url
DEFAULT_MIXKIT_URL_C = "https://mixkit.co/free-stock-music/"

# Incomplete url
MOOD_MIXKIT_URL_IC = "https://mixkit.co/free-stock-music/mood/"

MOODS = {
    "happy": "Happy",
    "sad": "Sad",
    "mellow": "Mellow",
    "aggresive": "Aggresive",
    "uplifting": "Uplifting",
    "whimsical": "Whimsical",
    "dramatic": "Dramatic",
    "eerie": "Eerie",
    "mysterious": "Mysterious",
    "romantic": "Romantic",
    "smooth": "Smooth",
    "upbeat": "Upbeat",
}


def load_soup(url):
    response = requests.get(url)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def get_rndm_mixkit_rfmp3_link(type=None):
    query_url = None
    if type == None:
        query_url = DEFAULT_MIXKIT_URL_C
    else:
        query_url = MOOD_MIXKIT_URL_IC + MOODS[type] + "/"
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
    os.makedirs("rfm/music", exist_ok=True)
    path = "rfm/music/" + name
    with open(path, "wb") as f:
        f.write(response.content)
    return path

def get_music_path():
    url = get_rndm_mixkit_rfmp3_link(random.choice(["happy", "mellow"]))
    return download_rfm_mp3(url)

if __name__ == "__main__":
    url = get_rndm_mixkit_rfmp3_link(random.choice(["happy", "mellow"]))
    download_rfm_mp3(url)
    print("Downloaded RFM mp3")
