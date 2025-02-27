import json
import random
import shutil
import sys
import numpy as np
import requests
from groq import Groq
import os, time
from PIL import Image
from moviepy import *
import threading
from moviepy import *

from multiprocessing import Pool

import edge_tts

VOICE = "en-AU-WilliamNeural"
RATE = "+20%"
PITCH = "+20Hz"

sys.path.append("rfm")
from rfm import get_music_path


T_WIDTH, T_HEIGHT = 1080, 1920

FPS = 30


GROQCLIENT = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

GENERATE_CAPTION_PROMPT = """Analyze the provided image and identify the captions that are part of the meme's intended dialogue or message. You should not have to describe the meme format/image as this is a OCR tool. Only the relevant text on the image has to be processed. Ignore any watermarks, credits, or unrelated text. Do not duplicate any captions. If there is no text, return an empty list of reading_order. Determine the natural reading order as a human would perceive it. Then, output the captions in JSON format with an ordered list, as follows:
        {
  "reading_order": [
    "First caption",
    "Second caption",
  ]
}
Ensure the order reflects logical reading patterns based on spatial positioning and dialogue structure."
"""


def get_text_reply(prompt, _model="qwen-2.5-32b"):
    print("[GROQ] Sending request to Groq API...")
    try:
        chat_completion = GROQCLIENT.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=_model,
        )
        print("[GROQ] Received response from Groq API")
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"[GROQ] Error in API call: {str(e)}")

        if "Please try again in " in str(e):
            wt = str(e).split("Please try again in ")[1].split(". ")[0].split("m")
            wait_time = float(wt[0]) * 60 + float(wt[1].split("s")[0])
            print(f"[DEBUG] Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
        return get_text_reply(prompt)


def get_image_reply(text, image_url, _model="llama-3.2-11b-vision-preview"):
    print("[GROQ] Sending request to Groq API...")
    try:
        image_completion = GROQCLIENT.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                            + " You have to return the result in json format.",
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            model=_model,
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        print("[GROQ] Received response from Groq API")

        message = image_completion.choices[0].message.content

        message = json.loads(message)

        return message
    except Exception as e:
        print(f"[GROQ] Error in API call: {str(e)}")

        if "Please try again in " in str(e):
            wt = str(e).split("Please try again in ")[1].split(". ")[0].split("m")
            wait_time = float(wt[0]) * 60 + float(wt[1].split("s")[0])
            print(f"[DEBUG] Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
        return get_image_reply(text, image_url)


def get_reddit_posts(subreddit="memes", type="hot", time_frame="day"):
    url = f"https://www.reddit.com/r/{subreddit}/{type}/.json"

    if type == "top":
        url += f"?t={time_frame}"

    headers = {
        "User-Agent": "Chrome/133.0.6943.98",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def get_meme_urls(posts: list):
    meme_urls: list = []
    for post in posts["data"]["children"]:
        if post["data"].get("url_overridden_by_dest") is None:
            continue
        if post["data"]["url_overridden_by_dest"].endswith((".jpeg", ".png")):
            if post["data"]["ups"] > MIN_UPVOTES:
                meme_urls.append(
                    (post["data"]["url_overridden_by_dest"], post["data"]["author"])
                )
    random.shuffle(meme_urls)
    return meme_urls


def generate_audio(text, idx):
    os.makedirs("redditmeme2video/audio", exist_ok=True)
    
    output_file = f"redditmeme2video/audio/audio_{idx}.mp3"

    tts = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH)
    tts.save_sync(output_file)

    return output_file


def download_image(url, idx):
    """Download meme image from URL and save locally"""
    response = requests.get(url)
    if response.status_code == 200:
        filename = f"redditmeme2video/images/meme_{idx}.png"
        os.makedirs("redditmeme2video/images", exist_ok=True)
        with open(filename, "wb") as f:
            f.write(response.content)
        return filename
    return None


def collect_captions_for_video(subreddit: str, amt=3, type="hot", retry=False):
    """
    Collect meme URLs and captions for one subreddit.
    Returns a tuple (meme_urls, pre_captions).
    """
    posts = get_reddit_posts(subreddit=subreddit, type=type, time_frame="week" if retry else "day")
    meme_urls = get_meme_urls(posts)
    if len(meme_urls) < amt:
        print(f"[DEBUG] Not enough memes for {subreddit}")
        if retry:
            exit()
        print("[DEBUG] Retrying with weekly posts")
        return collect_captions_for_video(subreddit, amt=amt, type="top", retry=True)
    # Let user select once for this video.
    while True:
        new_meme_urls = random.sample(meme_urls, amt)
        print(f"\nCaptions will be generated for the following URLs in r/{subreddit}:")
        for url in new_meme_urls:
            print("    -", url[0])
        if input("Proceed with these memes? (y/n): ").lower() == "y":
            meme_urls = new_meme_urls
            break
    pre_captions = []
    for idx, meme_url in enumerate(meme_urls):
        image_reply = get_image_reply(GENERATE_CAPTION_PROMPT, meme_url[0])
        captions = image_reply["reading_order"]
        # Allow user checking once per meme.
        if not AUTO:
            print(f"\nImage URL: {meme_url[0]}")
            print("[DEBUG] Received captions:")
            for i, caption in enumerate(captions):
                print(f"    {i}: {caption}")
            while True:
                entry = input("For this image, enter index to remove or 'e' to edit (press Enter if ok): ")
                if entry == "":
                    break
                elif entry == "e":
                    _index = input("Enter the index to edit: ")
                    _caption = input("Enter the new caption: ")
                    captions[int(_index)] = _caption
                else:
                    try:
                        captions.pop(int(entry))
                    except:
                        print("Invalid index to remove")
        pre_captions.append(captions)
    return meme_urls, pre_captions

def generate_video_clip(image_url, idx=0, captions=None):
    # Removed caption generation and user interaction from here.
    if captions is None:
        # Fallback: obtain captions automatically without interaction.
        image_reply = get_image_reply(GENERATE_CAPTION_PROMPT, image_url)
        captions = image_reply["reading_order"]
    # Process captions: trim punctuation
    for i, caption in enumerate(captions):
        caption = caption.strip()
        if caption and caption[-1] in [".", ",", "!", "?"]:
            caption = caption[:-1]
        captions[i] = caption

    audio_path = generate_audio(". ".join(captions), idx)

    meme_path = download_image(image_url, idx)
    img = Image.open(meme_path)

    # Calculate resize ratio
    img_ratio = min(T_WIDTH * 0.95 / img.width, T_HEIGHT * 0.95 / img.height)
    new_size = (int(img.width * img_ratio), int(img.height * img_ratio))
    img = img.resize(new_size, Image.LANCZOS)

    # Create audio clip
    audio_clip = AudioFileClip(audio_path)

    # Create meme clip
    meme_bg = Image.new("RGBA", (T_WIDTH, T_HEIGHT), (0, 0, 0, 0))
    paste_x = (T_WIDTH - new_size[0]) // 2
    paste_y = (T_HEIGHT - new_size[1]) // 2
    meme_bg.paste(img, (paste_x, paste_y))

    meme_clip = ImageClip(np.array(meme_bg)).with_duration(audio_clip.duration)

    # Create Video with effects based on idx
    if idx == 0:
        final_clip = CompositeVideoClip(
            [meme_clip.with_effects([vfx.FadeOut(0.05)])]
        ).with_audio(audio_clip)
    else:
        final_clip = CompositeVideoClip(
            [meme_clip.with_effects([vfx.FadeIn(0.05), vfx.FadeOut(0.05)])]
        ).with_audio(audio_clip)

    return final_clip


def write_video_to_file(video_clip: VideoClip, output_path):
    """Write video to file in a separate thread"""
    video_clip.write_videofile(
        output_path,
        codec="libx264",
        fps=FPS,
        threads=8,
        temp_audiofile_path = output_path[:output_path.rfind("/")],
    )

def generate_final_video(
    subreddit: str, amt=3, type="hot", retry=False, comments=False, location="redditmeme2video/output",
    pre_meme_urls=None, pre_captions=None
):
    comment = "Credits to"
    description = f"Compilation of {amt} {subreddit} memes. Enjoy!"
    if pre_meme_urls is None or pre_captions is None:
        # Fallback to internal collection (if needed interactively)
        pre_meme_urls, pre_captions = collect_captions_for_video(subreddit, amt, type, retry)
    title = get_image_reply(f"""
Given the following image:
    
Generate a social media package that includes:
- Title: A funny interesting, maybe exaggerated or controversial, title for the meme (make sure you understand the meme). Add tags at the end. Make sure it is not longer than 80 characters.

Return the result in JSON format:
{{
    "title": "...",
}}
    """, pre_meme_urls[0][0])["title"] + f' | r/{subreddit}'
    print(f"[TITLE] {title}")
    clips = []
    for idx, meme_url in enumerate(pre_meme_urls):
        subclip = generate_video_clip(meme_url[0], idx, captions=pre_captions[idx])
        if idx == len(pre_meme_urls) - 1:
            subclip = subclip.with_effects([vfx.CrossFadeOut(0.5)])
        clips.append(subclip)
        comment += f" u/{meme_url[1]},"
    comment = comment[:-1] + " for the memes!"
    outro = VideoFileClip("redditmeme2video/outro.mp4").with_fps(FPS)
    clips.append(outro)
    final_video = concatenate_videoclips(clips)
    # ...existing code for background video processing...
    gameplay_dir = "redditmeme2video/gameplay"
    gameplay_files = [f for f in os.listdir(gameplay_dir) if f.endswith((".mp4", ".webm"))]
    if not gameplay_files:
        raise Exception("No gameplay videos found in directory")
    BOTTOM_CLIP_PATH = os.path.join(gameplay_dir, random.choice(gameplay_files))
    background_source = VideoFileClip(BOTTOM_CLIP_PATH).with_fps(FPS)
    print(f"[DEBUG] Original background video size: {background_source.size}")
    start = random.randrange(0, int(background_source.duration - final_video.duration))
    bg_clip = background_source.subclipped(start, start + final_video.duration)
    print(f"[DEBUG] Background clip original size: {bg_clip.size}")
    bg_clip = bg_clip.resized(height=T_HEIGHT)
    print(f"[DEBUG] Background after height resize: {bg_clip.size}")
    bg_clip = bg_clip.cropped(
        width=T_WIDTH,
        height=T_HEIGHT,
        x_center=bg_clip.size[0] // 2,
        y_center=bg_clip.size[1] // 2,
    )
    print(f"[DEBUG] Background after cropping: {bg_clip.size}")
    bg_audio_path = get_music_path()
    bg_audio_clip: AudioFileClip = (
        AudioFileClip(bg_audio_path)
        .subclipped(0, final_video.duration)
        .with_volume_scaled(0.05)
    )
    transparent_background = (
        ColorClip((T_WIDTH, T_HEIGHT), color=(0, 0, 0, 0))
        .with_duration(final_video.duration)
        .with_audio(bg_audio_clip)
    )
    final_video = CompositeVideoClip(
        [
            bg_clip.with_effects([vfx.MultiplyColor(1)]),
            final_video,
            transparent_background,
        ]
    )
    os.makedirs(location, exist_ok=True)
    output_path = f"{location}/final_video.mp4"
    write_thread = threading.Thread(
        target=write_video_to_file,
        args=(final_video, output_path)
    )
    write_thread.start()
    if comments:
        return {
            output_path: {
                "title": title,
                "description": description,
                "pinned_comment": comment,
            }
        }
    return {
        output_path: {
            "title": title,
            "description": description,
        }
    }


AUTO = False
DAY = 19
UPLOAD = False

NOW = False

SUBREDDITS = [
    # "MinecraftMemes",
    "HistoryMemes",
]

SUBREDDITSLIST = [
    "Animemes",
    "memes",
    "wholesomememes",
    "memes",
    "HistoryMemes",
    "RelationshipMemes",
]

MIN_UPVOTES = 1000  # For RelationshipMemes
# MIN_UPVOTES = 3000


if __name__ == "__main__":
    # First, pre-generate captions for all videos.
    precomputed = {}
    for idx, subreddit in enumerate(SUBREDDITS):
        print(f"\n--- Caption collection for r/{subreddit} ---")
        meme_urls, pre_captions = collect_captions_for_video(subreddit, amt=3, type="top")
        precomputed[idx] = (meme_urls, pre_captions)
    # Now render videos unattended using precomputed captions.
    for idx, subreddit in enumerate(SUBREDDITS):
        meme_urls, pre_captions = precomputed[idx]
        video_data = generate_final_video(subreddit, amt=3, type="top", comments=True,
                                          location=f"redditmeme2video/output/{idx}",
                                          pre_meme_urls=meme_urls, pre_captions=pre_captions)
        with open(f"redditmeme2video/output/{idx}/data.json", "w") as f:
            json.dump(video_data, f)
        print(video_data)
        # Clean up .mp3 files in rfm folders
        for folder in ['rfm', 'rfm/music', 'redditmeme2video/audio', 'redditmeme2video/images']:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if file.endswith(['.mp3', '.png']):
                        os.remove(os.path.join(folder, file))
    if NOW:
        video_data = generate_final_video("memes", amt=3, type="top", comments=True)
