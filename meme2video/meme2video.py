import json
import random
import requests
from groq import Groq
import os, time
from moviepy import *
import numpy as np
from PIL import Image, ImageDraw
import requests
import sys

sys.path.append("ytupload")
from ytupload import main as uploader
from ytupload import create_schedule_time as yttime
from ytupload import init_file as ytinit

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

ytinit()


def get_text_reply(prompt, _model="qwen-2.5-32b"):
    print("[DEBUG] Sending request to Groq API...")
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=_model,
        )
        print("[DEBUG] Received response from Groq API")
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"[DEBUG] Error in API call: {str(e)}")

        if "Please try again in " in str(e):
            wt = str(e).split("Please try again in ")[1].split(". ")[0].split("m")
            wait_time = float(wt[0]) * 60 + float(wt[1].split("s")[0])
            print(f"[DEBUG] Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
        return get_text_reply(prompt)


def get_top_memes():
    url = "https://api.imgflip.com/get_memes"
    headers = {"User-agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    data = response.json()
    memes = data["data"]["memes"]
    return memes


def generate_single_meme_content(meme_data):
    template_name = meme_data["name"]
    num_tb = meme_data["box_count"]
    prompt = f"""Generate an extremely funny and clever caption for the '{template_name}' meme template.
    The caption must be similar to its typical usage.
    Create {num_tb} lines of text, each no more than 10 words.
    Make it hilarious, witty, and somewhat unexpected. Aim for humor that's relatable, potentially viral, and appeals to a wide audience.
    Incorporate current trends, pop culture references, or universal experiences that people can relate to.
    Use clever wordplay, puns, or maybe subvert expectations for maximum comedic effect.
    Ensure the humor is appropriate for a general audience.
    Aim for a punchline that's both surprising and satisfying.
    Do not give any explanation or context, just the caption.
    Separate the lines with a '|' character. There should be no use of emojis or any chinese characters. Only English alphabets allowed. If there are 3 lines, then ensure that the first line is the main caption"""
    caption = get_text_reply(prompt)
    print(f"[DEBUG] Caption: {caption}")

    if "template:" in caption:
        caption = caption.split("template:")[1].strip()

    captions = caption.split("|")
    if "|" not in caption:
        captions = caption.split("\n")
    for idx, cap in enumerate(reversed(captions)):
        if cap.strip() == "":
            print(f"Error: Caption {idx+1} is empty")
            captions.remove(cap)

    if len(captions) == num_tb:
        return {
            "template_name": template_name,
            "captions": captions,
            "meme_data": meme_data,
        }
    else:
        print(f"Error: Expected {num_tb} lines of text, but received {len(captions)}")
        return generate_single_meme_content(meme_data)


def evaluate_single_meme(template_name, captions):
    prompt = f"""Rate the following caption for the '{template_name}' meme template on a scale of 1-10 for humor, and relatability:
    {' | '.join(captions)}
    Provide only a averaged final numeric score, no explanation."""

    response = get_text_reply(prompt, "gemma2-9b-it")
    print(f"[REPORT] Evaluation: '{response}'")
    try:
        return float(response.strip())
    except ValueError:
        return 0


def create_meme_image(meme_data, captions):
    url = "https://api.imgflip.com/caption_image"
    headers = {"User-agent": "Mozilla/5.0"}

    params = {
        "template_id": meme_data["id"],
        "username": "Ryash23",
        "password": "yash2327raj",
    }

    for i, caption in enumerate(captions):
        params[f"boxes[{i}][text]"] = caption

    response = requests.post(url, headers=headers, data=params)
    data = response.json()
    if data["success"]:
        return data["data"]["url"]
    else:
        print(f"[ERROR] Error creating meme: {data['error_message']}")
        return None


def generate_memes(start):
    memes_templates = [random.choice(get_top_memes()) for _ in range(start)]
    # print(f"Templates: {memes_templates}")
    url_list = []
    for idx, template in enumerate(memes_templates):
        print(f"[DEBUG] Template: {template}")
        i = 0
        meme_caption = None
        while True:
            print(f"[DEBUG] Attempt: {i}")
            meme_caption = generate_single_meme_content(template)
            print(f"[DEBUG] Template: {meme_caption}")
            score = evaluate_single_meme(
                meme_caption["template_name"], meme_caption["captions"]
            )
            print(f"[DEBUG] Score: {score}")

            SCORE = 7.5
            if score >= SCORE:

                cap = meme_caption["captions"]
                if cap[0].strip(' "?-,').lower() == "change my mind":
                    cap.pop(0)

                url = create_meme_image(meme_caption["meme_data"], cap)
                print(f"[URL] URL: {url}")

                image_download = download_meme(url, idx)
                _img = Image.open(image_download)
                _img.show()

                if input(f"Approved {i}? (y/n): ") == "n":
                    i = 0
                    template = random.choice(get_top_memes())
                    continue

                url_list.append({"url": url, "caption": meme_caption["captions"]})
                break
            else:
                print(f"[DEBUG] Score is less than {SCORE}, retrying...")


            i += 1

            if i >= 25:
                i = 0
                print(f"[DEBUG] Too many attempts, changing template...")
                template = random.choice(get_top_memes())
                continue

    return url_list


def download_meme(url, idx):
    """Download meme image from URL and save locally"""
    response = requests.get(url)
    if response.status_code == 200:
        filename = f"meme2video/images/meme_{idx}.png"
        os.makedirs("meme2video/images", exist_ok=True)
        with open(filename, "wb") as f:
            f.write(response.content)
        return filename
    return None


def make_video(url_list):
    """Create a video from list of meme URLs with MC parkour as background"""
    target_width, target_height = 1080, 1920
    print(f"[DEBUG] Target dimensions: {target_width}x{target_height}")
    FPS = 24

    video_clips = []

    # for idx, url in enumerate(url_list):
    #     download_meme(url["url"], idx)

    # if input("Do you want to continue? (y/n): ") == "n":
    #     exit()

    for idx, url in enumerate(url_list):
        print(f"\n[DEBUG] Processing meme {idx+1}/{len(url_list)}")

        # Download meme image
        meme_path = download_meme(url["url"], idx)
        if not meme_path:
            continue

        # Load and check meme image dimensions
        img = Image.open(meme_path)
        print(f"[DEBUG] Original meme image size: {img.size}")

        # Calculate resize ratio
        img_ratio = min(
            target_width * 0.95 / img.width, target_height * 0.95 / img.height
        )
        new_size = (int(img.width * img_ratio), int(img.height * img_ratio))
        print(f"[DEBUG] Resized meme dimensions: {new_size}")
        img = img.resize(new_size, Image.LANCZOS)

        # Generate and process audio
        audio_path = f"meme2video/audio/meme_{idx}.mp3"
        os.makedirs("meme2video/audio", exist_ok=True)

        # Get text from meme to narrate
        meme_text = ". ".join([caption.strip() for caption in url["caption"]])

        # Generate audio using UnrealSpeech API
        response = requests.post(
            "https://api.v7.unrealspeech.com/stream",
            headers={
                "Authorization": "Bearer ja2oPEHnKZkJpYU1EPB3IiMnZg3RxABWgFGwQHtAnz92qQL66NWQKE"
            },
            json={
                "Text": meme_text,
                "VoiceId": "Dan",
                "Bitrate": "192k",
                "Speed": "0.4",
                "Pitch": "0.92",
                "Codec": "libmp3lame",
            },
        )
        with open(audio_path, "wb") as f:
            f.write(response.content)

        # Create audio clip
        audio_clip = AudioFileClip(audio_path)
        print(f"[DEBUG] Audio duration: {audio_clip.duration}s")

        # Create meme clip
        meme_bg = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
        paste_x = (target_width - new_size[0]) // 2
        paste_y = (target_height - new_size[1]) // 2
        print(f"[DEBUG] Meme paste position: ({paste_x}, {paste_y})")
        meme_bg.paste(img, (paste_x, paste_y))

        meme_clip = ImageClip(np.array(meme_bg)).with_duration(audio_clip.duration)
        print(f"[DEBUG] Meme clip size: {meme_clip.size}")

        # Create composite
        final_clip = None
        if idx == 0:
            final_clip = CompositeVideoClip(
                [meme_clip.with_effects([vfx.FadeOut(0.05)])]
            ).with_audio(audio_clip)
        else:
            final_clip = CompositeVideoClip(
                [meme_clip.with_effects([vfx.FadeIn(0.05), vfx.FadeOut(0.05)])]
            ).with_audio(audio_clip)

        print(f"[DEBUG] Final composite clip size: {final_clip.size}\n")

        video_clips.append(final_clip)

    # Create final video
    final_video = concatenate_videoclips(video_clips)

    # Load background video
    BOTTOM_CLIP_PATH = "reddit2image/gameplay/mcparkour.mp4"
    background_source = VideoFileClip(BOTTOM_CLIP_PATH)
    print(f"[DEBUG] Original background video size: {background_source.size}")

    # Process background video section
    start = random.randrange(0, int(background_source.duration - final_video.duration))
    bg_clip = background_source.subclipped(start, start + final_video.duration)
    print(f"[DEBUG] Background clip original size: {bg_clip.size}")

    # Resize background
    bg_clip = bg_clip.resized(height=target_height)
    print(f"[DEBUG] Background after height resize: {bg_clip.size}")

    bg_clip = bg_clip.cropped(
        width=target_width,
        height=target_height,
        x_center=bg_clip.size[0] // 2,
        y_center=bg_clip.size[1] // 2,
    )
    print(f"[DEBUG] Background after cropping: {bg_clip.size}")

    final_video = CompositeVideoClip(
        [bg_clip.with_effects([vfx.MultiplyColor(0.7)]), final_video]
    )

    print(f"[DEBUG] Final concatenated video size: {final_video.size}")

    # Save video
    output_path = "meme2video/output/meme_video.mp4"
    os.makedirs("meme2video/output", exist_ok=True)
    final_video.write_videofile(output_path, fps=FPS)

    # Cleanup
    for idx in range(len(url_list)):
        try:
            os.remove(f"meme2video/images/meme_{idx}.png")
            os.remove(f"meme2video/audio/meme_{idx}.mp3")
        except:
            pass

    return output_path


def get_social_content(caption):
    """
    Generate a social media package in english: title, and description for the given meme.
    """
    social_prompt = f"""
Given the following story:
Meme: {' | '.join(caption)}

Generate a social media package in english that includes:
- Title: A humorous but meaningful title for the content.
- Description: A concise description suitable for a video description. There should be numerous related tags as well here. There should be popular common tags here as well.
- Pinned Comment: A friendly and engaging related first comment to pin.

Return the result in english in JSON format:
{{
    "title": "...",
    "description": "...",
    "pinned_comment": "..."
}}
    """
    response = get_text_reply(social_prompt)
    try:
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        json_str = response[start_idx:end_idx]
        return json.loads(json_str)
    except Exception as e:
        print(f"[ERROR] Failed to parse social content JSON: {e}")
        print(f"[ERROR] Retrying...")
        get_social_content(caption)


if __name__ == "__main__":

    schedules = [
        yttime(year=2025, month=2, day=16, hour=23, minute=59, tz_offset=4),
        # yttime(year=2025, month=2, day=16, hour=8, minute=0, tz_offset=4),
    ]

    for sce in schedules:
        urls = generate_memes(3)
        social_content = get_social_content(urls[0]["caption"])
        video_path = make_video(urls)
        print(f"[SOCIAL CONTENT] { {video_path: social_content} }")

        uploader({video_path: social_content}, sce)

        print(f"[FINAL VIDEO] {video_path}")
