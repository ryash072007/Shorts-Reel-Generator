from groq import Groq
import json, os, sys, time, random
from moviepy import *
from TTS.api import TTS
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np

sys.path.append("background")
from background import get_fast_images, quit_browser

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch", progress_bar=True, gpu=False)
text_color = (255, 255, 255, 255)  # White text for better visibility

# Add new constants
FONT_SIZE = 140  # Increased base font size
FONT_PATH = "impact.ttf"  # Change to your preferred font
TEXT_SHADOW_COLOR = (0, 0, 0, 255)
TEXT_BACKGROUND_OPACITY = 0  # No background for better readability
WORD_FADE_DURATION = 0.001
SCALE_RANGE = (0.9, 1.1)


def generate_story(emotion, additional_prompt=""):
    print(f"[DEBUG] Generating story with emotion: {emotion}")
    if additional_prompt:
        print(f"[DEBUG] Additional prompt: {additional_prompt}")
    prompt = f"""Create a compelling short story (50-100 words) that primarily evokes the emotion of {emotion}. 
    Additional requirements: {additional_prompt}
    
    The story should be vivid, engaging, and suitable for visual storytelling.
    Focus on descriptive language and emotional depth.
    
    Return the story in a simple text format."""

    result = get_text_reply(prompt)
    print(f"[DEBUG] Story generated successfully ({len(result)} characters)")
    return result


def break_down_story(story):
    print("[DEBUG] Breaking down story into segments...")
    prompt = f"""Break down this story into 2 sensible and related segments. For each segment:
    1. Create expressive SSML-formatted text for narration
    2. Generate a detailed image prompt that captures the scene's emotion
    3. Extract the raw text for captions

    Format as JSON:
    {{
        "segments": [
            {{
                "narration": "[SSML formatted text]",
                "image_prompt": "[Detailed scene description]",
                "raw_text": "[Plain text for captions]"
            }}
        ]
    }}

    Story: {story}"""

    response = get_text_reply(prompt)
    print("[DEBUG] Story breakdown complete")
    return_dict = convert_output_to_dict(response)
    if not return_dict:
        print("[ERROR] Failed to parse story segments, retrying...")
        return break_down_story(story)
    return return_dict


def get_text_reply(prompt):
    print("[DEBUG] Sending request to Groq API...")
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
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
    word_duration = duration / len(words) if words else duration

    def make_frame(t):
        current_index = int(t // word_duration)
        word = words[min(current_index, len(words) - 1)]

        # Calculate scale effect
        word_start_time = current_index * word_duration
        word_progress = (t - word_start_time) / word_duration

        # Scale effect
        progress_scale = min(1.0, word_progress * 2)
        scale = SCALE_RANGE[0] + (SCALE_RANGE[1] - SCALE_RANGE[0]) * progress_scale

        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(FONT_PATH, int(font_size * scale))
        except:
            font = ImageFont.load_default()
            print(f"[DEBUG] Fallback to default font - couldn't load {FONT_PATH}")

        bbox = draw.textbbox((0, 0), word, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (width - w) // 2
        y = (height - h) // 2

        padding = 20
        background_bbox = (x - padding, y - padding, x + w + padding, y + h + padding)
        draw.rectangle(background_bbox, fill=(0, 0, 0, TEXT_BACKGROUND_OPACITY))
        # Set opacity instantly to full (no fade effect)
        text_color_with_opacity = (*text_color[:3], 255)
        draw.text((x, y), word, font=font, fill=text_color_with_opacity)

        return np.array(img)

    return VideoClip(make_frame, duration=duration)


from concurrent.futures import ThreadPoolExecutor


def generate_audio_for_segment(segment, idx):
    audio_path = f"AiStoryMaker/audio/segment_{idx}.mp3"
    tts.tts_to_file(text=segment["narration"], file_path=audio_path, speed=1.5)
    print(f"[DEBUG] Audio generated for segment {idx+1} at {audio_path}")
    return audio_path


def create_video(segments):
    print(f"[DEBUG] Creating enhanced video with {len(segments)} segments")
    video_clips = []

    # Generate audio clips concurrently
    audio_paths = [None] * len(segments)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(generate_audio_for_segment, seg, idx): idx
            for idx, seg in enumerate(segments)
        }
        for future in futures:
            idx = futures[future]
            audio_paths[idx] = future.result()

    for idx, segment in enumerate(segments):
        print(f"[DEBUG] Processing segment {idx+1}/{len(segments)}")
        audio_path = audio_paths[idx]
        audio_clip = AudioFileClip(audio_path)

        get_fast_images([segment["image_prompt"]], prefix="", start=idx)
        image_path = f"background/images/{idx}.png"

        img = Image.open(image_path)
        target_width, target_height = 1080, 1920
        half_height = target_height // 2

        # Create transparent background for full frame
        background_clip = ColorClip(size=(target_width, target_height), color=(0, 0, 0, 0)).with_duration(audio_clip.duration)

        # Resize the full image to fit the top half without cropping
        img_top = img.resize((target_width, half_height), Image.LANCZOS)

        # Apply a gradient and blur effect on the resized image
        gradient = Image.new("RGBA", (target_width, half_height), (0, 0, 0, 0))
        gradient_draw = ImageDraw.Draw(gradient)
        gradient_draw.rectangle((0, 0, target_width, half_height), fill=(0, 0, 0, 80))
        img_top = img_top.filter(ImageFilter.GaussianBlur(radius=2))
        img_top = Image.alpha_composite(img_top.convert("RGBA"), gradient)
        base_clip_top = ImageClip(np.array(img_top)).with_duration(audio_clip.duration)

        # Create dynamic text clip sized and centered to the top half
        text_clip = create_dynamic_text_clip(
            segment["raw_text"], audio_clip.duration, target_width, half_height
        ).with_position(("center", half_height//2))

        # Composite clips: black background, top image, and text
        final_clip = CompositeVideoClip([
            background_clip,
            base_clip_top.with_position(("center", "top")),
            text_clip
        ], size=(target_width, target_height)).with_audio(audio_clip)

        final_clip = final_clip.with_effects(
            [vfx.FadeIn(0.1), vfx.FadeOut(0.1)]
        )
        video_clips.append(final_clip)
        print(f"[DEBUG] Enhanced segment {idx+1} complete")

    quit_browser()

    # Concatenate story segments
    story_video = concatenate_videoclips(video_clips, method="compose")
    
    # Append the external bottom video clip
    BOTTOM_CLIP_PATH = "reddit2image/gameplay/mcparkour - FPS - Videobolt.net.mp4"  # path to your external bottom video
    bottom_source = VideoFileClip(BOTTOM_CLIP_PATH).with_fps(4).with_duration(story_video.duration)
    # Crop the external video: take a vertical slice of height equal to half_height from the center
    crop_y1 = int((bottom_source.h - half_height) / 2)
    crop_y2 = crop_y1 + half_height
    crop_x1 = int((bottom_source.w - target_width) / 2)
    crop_x2 = crop_x1 + target_width
    bottom_cropped = bottom_source.cropped(y1=crop_y1, y2=crop_y2, x1=crop_x1, x2=crop_x2)
    # Resize the cropped video to fill the target width and bottom half height
    # bottom_resized = bottom_cropped.resized(new_size=(target_width, half_height))
    # Create black background for full frame and overlay the bottom video at bottom
    bottom_bg = ColorClip(size=(target_width, target_height), color=(0, 0, 0, 0)) #.with_duration(bottom_resized.duration)
    bottom_composite = CompositeVideoClip([
        bottom_bg,
        bottom_cropped.with_position(("center", "bottom"))
    ], size=(target_width, target_height), is_mask=True).with_duration(story_video.duration)
    
    # Concatenate story video with bottom composite for continuous playback
    final_video = concatenate_videoclips([story_video, bottom_composite], method="compose")
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


def main(emotion, additional_prompt="", only_descriptor=False):
    print("[DEBUG] Starting AiStoryMaker...")
    print(f"[DEBUG] Creating directories...")
    # Create necessary directories
    os.makedirs("AiStoryMaker/audio", exist_ok=True)
    os.makedirs("AiStoryMaker/output", exist_ok=True)

    print("[DEBUG] Story generation started")
    print("Generating story...")
    story = generate_story(emotion, additional_prompt)

    descriptors = get_social_content(story)
    print(f"[DEBUG] Social media package generated: {descriptors}")

    if only_descriptor:
        exit()

    print("[DEBUG] Story breakdown started")
    print("Breaking down story into segments...")
    story_data = break_down_story(story)
    print(f"[DEBUG] Story broken down into {len(story_data['segments'])} segments")

    print("[DEBUG] Video creation started")
    print("Creating video...")
    final_video = create_video(story_data["segments"])

    print("[DEBUG] Saving final video...")
    print("Saving video...")
    output_path = f"AiStoryMaker/output/output.mp4"
    final_video.write_videofile(output_path, fps=4)

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

    return {"video_path": output_path, "social_content": descriptors}


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
    print("[DEBUG] AiStoryMaker script started")
    emotion = (
        random.choice(emotions)
        if random.random() < 0.5
        else random.choice(emotions) + " and " + random.choice(emotions)
    )
    additional_prompt = generate_unique_additional_prompt(emotion)
    details = main(emotion, additional_prompt)
    print(details)


if __name__ == "__main__":
    # print("[DEBUG] AiStoryMaker script started")
    # emotion = input("Enter the primary emotion for the story: ")
    # additional_prompt = input("Enter any additional requirements (or press Enter to skip): ")
    # main(emotion, additional_prompt, True)
    for i in range(1):
        GenerateSinglePost()
