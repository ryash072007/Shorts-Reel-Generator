from groq import Groq
import json, os, sys, time, random
from moviepy import *

# from TTS.api import TTS
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np
import requests

sys.path.append("background")
sys.path.append("ytupload")
from background import get_fast_images, quit_browser, set_intialised
from ytupload import main as uploader
from ytupload import init_file as ytinit

import os


# clientV = Client(
#     user_id=os.getenv("PLAY_HT_USER_ID"),
#     api_key=os.getenv("PLAY_HT_API_KEY"),
# )

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch", progress_bar=True, gpu=False)
text_color = (255, 255, 255, 255)  # White text for better visibility

# Add new constants
FONT_SIZE = 100  # Reduced from 140
FONT_PATH = "impact.ttf"  # Change to your preferred font
TEXT_SHADOW_COLOR = (0, 0, 0, 255)
TEXT_BACKGROUND_OPACITY = 0  # No background for better readability
WORD_FADE_DURATION = 0.001
SCALE_RANGE = (0.9, 1.1)
FPS = 24


def generate_story():
    print("[DEBUG] Generating story...")
    prompt = """Write a unique (75 word), funny darkly humorous story in the form of a paragraph. Start with a strong hook to grab attention within the first few seconds. The criteria for judgment are "Strong Setup and Punchline", "Witty Dialogue", "Absurdity", "Irony and Satire", "Dark Twist". Use simple names, if any. """
    result = get_text_reply(prompt)
    print(f"[DEBUG] Story generated successfully ({len(result)} characters)")

    grade = grade_story(result)
    print(f"[DEBUG] Story grade is {grade}")
    if grade < 9:
        print("[DEBUG] Story grade is below 8, retrying...")
        return generate_story()

    return result


def grade_story(story):
    print("[DEBUG] Grading story...")
    prompt = f"""
Grade the following on a scale from 1 to 10 based on these criteria (IMPORTANT: ensure that only the final score out of 10 is produced as output): "Strong Setup and Punchline", "Witty Dialogue", "Absurdity", "Irony and Satire", "Dark Twist". The Story: '{story}' ```Example Output: [Grade as a number]```"
"""
    response = get_text_reply(prompt, _model="deepseek-r1-distill-llama-70b")
    try:
        return int(response[response.rfind("\n") :].strip("* []"))
    except:
        print("[ERROR] Failed to parse grade, retrying...")
        return grade_story(story)


def break_down_story(story):
    print("[DEBUG] Breaking down story into segments...")
    prompt = f"""Break down this story into 7-9 sensible and related segments. For each segment:
    1. Generate a detailed image prompt that captures the scene or emotion.
    2. Extract the raw text for captions. The sum of all raw texts should exactly be the entire story.

    Format as JSON:
    {{
        "segments": [
            {{
                "image_prompt": "[Detailed scene description]",
                "raw_text": "[Plain text for captions]"
            }}
        ]
    }}

    Story: {story}"""

    response = get_text_reply(prompt)
    print(f"[DEBUG] Story breakdown complete, breakdown: {response}")
    return_dict = convert_output_to_dict(response)
    if not return_dict:
        print("[ERROR] Failed to parse story segments, retrying...")
        return break_down_story(story)
    return return_dict


def get_text_reply(prompt, _model="llama-3.3-70b-versatile"):
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
        print("[DEBUG] Retrying after 5 seconds...")
        time.sleep(5)
        return get_text_reply(prompt)


def convert_output_to_dict(output):
    try:
        start_idx = output.find("{")
        end_idx = output.rfind("}") + 1
        json_str = output[start_idx:end_idx]
        return json.loads(json_str)
    except:
        print("[ERROR] Failed to parse JSON, returning empty structure")
        return None


def create_dynamic_text_clip(text, duration, width, height, font_size=FONT_SIZE):
    print(f"[DEBUG] Creating enhanced dynamic text clip (duration: {duration}s)")
    words = text.split()
    # Calculate timing for groups of three words
    group_count = (len(words) + 2) // 3  # Ceiling division to handle all words
    group_duration = duration / group_count

    def make_frame(t):
        # Calculate which group of three words we're currently showing
        current_group = int(t // group_duration)
        group_progress = (t % group_duration) / group_duration
        
        # Get the current group of three words
        start_idx = current_group * 3
        word_group = words[start_idx:start_idx + 3]
        # Pad with empty strings if we don't have 3 words
        word_group.extend([""] * (3 - len(word_group)))
        
        # Create image
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(FONT_PATH, font_size)
        except:
            font = ImageFont.load_default()
            print(f"[DEBUG] Fallback to default font - couldn't load {FONT_PATH}")

        # Join words with spaces
        display_text = " ".join(word_group)
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), display_text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (width - w) // 2
        y = (height - h) // 2

        # Draw background
        padding = 20
        background_bbox = (x - padding, y - padding, x + w + padding, y + h + padding)
        draw.rectangle(background_bbox, fill=(0, 0, 0, TEXT_BACKGROUND_OPACITY))

        # Draw the full text in white first
        draw.text((x, y), display_text, font=font, fill=text_color)
        
        # Calculate how many words in the current group should be highlighted
        words_to_highlight = min(3, int(group_progress * 3) + 1)
        
        # Highlight the appropriate number of words
        current_x = x
        for i, word in enumerate(word_group):
            if not word:
                continue
                
            # Highlight completed words
            if i < words_to_highlight:
                draw.text((current_x, y), word, font=font, fill=(255, 255, 0, 255))
            
            # Add space after each word except the last
            current_x += draw.textlength(word + " ", font=font)

        return np.array(img)

    return VideoClip(make_frame, duration=duration)


from concurrent.futures import ThreadPoolExecutor


def generate_audio_for_segment(segment, idx):
    audio_path = f"AiStoryMaker/audio/segment_{idx}.mp3"
    # tts.tts_to_file(text=segment["narration"], file_path=audio_path, speed=1.5)
    print(f"[DEBUG] Audio generated for segment {idx+1} at {audio_path}")
    return audio_path


def create_video(segments):
    print(f"[DEBUG] Creating enhanced video with {len(segments)} segments")
    video_clips = []

    # Generate audio clips concurrently
    audio_paths = [None] * len(segments)

    for idx, seg in enumerate(segments):
        print(f"[DEBUG] Generating audio for segment {idx+1}/{len(segments)}")
        audio_path = f"AiStoryMaker/audio/segment_{idx}.mp3"
        audio_paths[idx] = audio_path

        response = requests.post(
            "https://api.v7.unrealspeech.com/stream",
            headers={
                "Authorization": "Bearer ja2oPEHnKZkJpYU1EPB3IiMnZg3RxABWgFGwQHtAnz92qQL66NWQKE"
            },
            json={
                "Text": seg["raw_text"],  # Up to 1000 characters
                "VoiceId": "Dan",  # Dan, Will, Scarlett, Liv, Amy
                "Bitrate": "192k",  # 320k, 256k, 192k, ...
                "Speed": "0.2",  # -1.0 to 1.0
                "Pitch": "0.92",  # -0.5 to 1.5
                "Codec": "libmp3lame",  # libmp3lame or pcm_mulaw
            },
        )
        with open(audio_path, "wb") as f:
            f.write(response.content)

    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     futures = {
    #         executor.submit(generate_audio_for_segment, seg, idx): idx
    #         for idx, seg in enumerate(segments)
    #     }
    #     for future in futures:
    #         idx = futures[future]
    #         audio_paths[idx] = future.result()

    target_width, target_height = 1080, 1920
    half_height = target_height // 2

    for idx, segment in enumerate(segments):
        print(f"[DEBUG] Processing segment {idx+1}/{len(segments)}")
        audio_path = audio_paths[idx]
        audio_clip = AudioFileClip(audio_path)

        get_fast_images([segment["image_prompt"]], prefix="", start=idx)
        image_path = f"background/images/{idx}.png"

        img = Image.open(image_path)

        # Create transparent background for full frame
        # background_clip = ColorClip(size=(target_width, target_height), color=(0, 0, 0, 0)).with_duration(audio_clip.duration)

        # Resize the full image to fit the top half without cropping
        img_top = img.resize((target_width, half_height), Image.LANCZOS)

        # Apply a gradient and blur effect on the resized image
        gradient = Image.new("RGBA", (target_width, half_height), (0, 0, 0, 0))
        gradient_draw = ImageDraw.Draw(gradient)
        gradient_draw.rectangle((0, 0, target_width, half_height), fill=(0, 0, 0, 80))
        # img_top = img_top.filter(ImageFilter.GaussianBlur(radius=2))
        img_top = Image.alpha_composite(img_top.convert("RGBA"), gradient)
        base_clip_top = ImageClip(np.array(img_top)).with_duration(audio_clip.duration)

        # Create dynamic text clip sized and centered to the top half
        text_clip = create_dynamic_text_clip(
            segment["raw_text"], audio_clip.duration, target_width, half_height
        ).with_position(("center", half_height // 3))

        # Composite clips: black background, top image, and text
        final_clip = CompositeVideoClip(
            [
                # background_clip,
                base_clip_top,  # .with_position(("center", "top")),
                text_clip,
            ],
            size=(target_width, half_height),
        ).with_audio(audio_clip)

        final_clip = final_clip.with_effects([vfx.FadeIn(0.1), vfx.FadeOut(0.1)])

        video_clips.append(final_clip)
        print(f"[DEBUG] Enhanced segment {idx+1} complete")

    quit_browser()

    # Concatenate story segments
    story_video = concatenate_videoclips(video_clips, method="compose")

    # Append the external bottom video clip
    BOTTOM_CLIP_PATH = (
        "reddit2image/gameplay/mcparkour.mp4"  # path to your external bottom video
    )
    bottom_source: VideoClip = VideoFileClip(BOTTOM_CLIP_PATH)

    start = random.randrange(0, int(bottom_source.duration - story_video.duration))
    end = start + story_video.duration

    bottom_source = bottom_source.with_fps(FPS).subclipped(start, end)

    bottom_source = bottom_source.resized(height=target_height)
    # Crop the external video: take a vertical slice of height equal to half_height from the center
    # crop_y1 = int((bottom_source.h - half_height) / 2)
    # crop_y2 = crop_y1 + half_height
    # crop_x1 = int((bottom_source.w - target_width) / 2)
    # crop_x2 = crop_x1 + target_width
    bottom_cropped: VideoClip = bottom_source.cropped(
        width=target_width,
        height=half_height,
        x_center=bottom_source.size[0] // 2,
        y_center=bottom_source.size[1] // 2,
    )

    print(bottom_source.size)
    print(bottom_cropped.size)
    print(final_clip.size)

    # y1=crop_y1, y2=crop_y2, x1=crop_x1, x2=crop_x2)

    # Resize the cropped video to fill the target width and bottom half height
    # bottom_resized = bottom_cropped.resized(new_size=(target_width, half_height))
    # Create black background for full frame and overlay the bottom video at bottom
    # bottom_bg = ColorClip(size=(target_width, target_height), color=(0, 0, 0, 0),).with_duration(story_video.duration)
    # bottom_composite = CompositeVideoClip([
    #     bottom_bg,
    #     bottom_cropped.with_position(("center", "bottom"))
    # ], size=(target_width, target_height), is_mask=True)

    # Concatenate story video with bottom composite for continuous playback
    # final_video = concatenate_videoclips([story_video, bottom_cropped.with_position("center", "bottom")], method="compose")

    final_video = clips_array([[story_video], [bottom_cropped]])

    return final_video


def get_social_content(story):
    """
    Generate a social media package: title, description, and pinned comment for the given prompt and story.
    """
    social_prompt = f"""
Given the following story:
Story: {story}

Generate a social media package that includes:
- Title: A compelling title for the content.
- Description: A concise description suitable for a video description. There should be numerous tags as well here.
- Pinned Comment: A friendly and engaging first comment to pin.

Return the result in JSON format:
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
        get_social_content(story)
        # return {"title": "", "description": "", "pinned_comment": ""}


def main(only_descriptor=False):
    print("[DEBUG] Starting AiStoryMaker...")
    print(f"[DEBUG] Creating directories...")
    # Create necessary directories
    os.makedirs("AiStoryMaker/audio", exist_ok=True)
    os.makedirs("AiStoryMaker/output", exist_ok=True)

    print("[DEBUG] Story generation started")
    print("Generating story...")
    story = generate_story()

    print("[DEBUG] Story: ", story)

    descriptors = get_social_content(story)
    print(f"[DEBUG] Social media package generated: {descriptors}")

    print("[DEBUG] Story breakdown started")
    print("Breaking down story into segments...")
    story_data = break_down_story(story)
    print(f"[DEBUG] Story broken down into {len(story_data['segments'])} segments")

    if only_descriptor:
        exit()

    print("[DEBUG] Video creation started")
    print("Creating video...")
    final_video = create_video(story_data["segments"])

    print("[DEBUG] Saving final video...")
    print("Saving video...")
    output_path = f"AiStoryMaker/output/output.mp4"
    final_video.write_videofile(output_path, fps=FPS)

    print("[DEBUG] Cleaning up temporary files...")
    print("Cleaning up temporary files...")
    cleanup_count = 0
    for file in os.listdir("AiStoryMaker/audio"):
        os.remove(f"AiStoryMaker/audio/{file}")
        cleanup_count += 1
    for file in os.listdir("background/images"):
        os.remove(f"background/images/{file}")
        cleanup_count += 1
    print(f"[DEBUG] Removed {cleanup_count} temporary files")

    print(f"Video saved to {output_path}")

    return {output_path: descriptors}


emotions = [
    "Euphoria",
    "Nostalgia",
    "Awe",
    "Satisfaction",
    "Serenity",
    "Anticipation",
    "Wholesomeness",
    "Triumph",
    "Surprise",
    "Empowerment",
    "Curiosity",
    "Hope",
    "Gratitude",
    "Vulnerability",
    "Longing",
    "Melancholy",
    "Regret",
    "Hilarity",
    "Shock",
    "Relief",
    "Adrenaline",
    "Inspiration",
    "Love",
    "Contentment",
    "Tension",
    "Compassion",
    "Playfulness",
    "Defeat",
    "Resilience",
    "Shockwave",
]


def generate_unique_additional_prompt(emotion):
    """
    Generate a unique additional prompt for the given emotion using Groq,
    ensuring the resulting story will be intriguing and fun for social media.
    """
    prompt = f"""
Given the emotion '{emotion}', generate an additional prompt, that when given to an AI, that would inspire a captivating, intriguing, and fun short complete story optimized for social media visuals. Provide creative details to inspire engaging storytelling.
Return just the text.
	"""
    return get_text_reply(prompt).strip()


def GenerateSinglePost():
    try:
        details = main()
        set_intialised(False)
        print("[DEBUG]:", details)
    except Exception as e:
        set_intialised(False)
        print(f"[ERROR]: {e}\nRetrying...")
        GenerateSinglePost()
        return

    uploader(details)


if __name__ == "__main__":
    # print("[DEBUG] AiStoryMaker script started")
    # emotion = input("Enter the primary emotion for the story: ")
    # additional_prompt = input("Enter any additional requirements (or press Enter to skip): ")
    # main(emotion, additional_prompt, True)
    print("[DEBUG] AiStoryMaker script started")
    ytinit()
    for i in range(16):
        print(f"[NOTICE] Generating post {i+1}")
        GenerateSinglePost()
