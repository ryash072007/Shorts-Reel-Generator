import textwrap
from groq import Groq
import json, os, sys, random, time
# from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, VideoClip, CompositeVideoClip
from moviepy import *
from TTS.api import TTS
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np

sys.path.append("background")
sys.path.append("reddit")

import background, reddit

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

tts = TTS(model_name="tts_models/en/jenny/jenny", progress_bar=True, gpu=False)

history = []

text_color = (0, 0, 0, 255)  # White by default, can be changed to any RGBA value

def get_text_reply(prompt, add_to_memory=False):
    global history

    try:
        chat_completion = client.chat.completions.create(
            messages=history
            + [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-70b-8192",
        )
    except Exception as e:
        time.sleep(120)
        return get_text_reply(prompt, add_to_memory)

    if add_to_memory:
        history.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        history.append(
            {"role": "system", "content": chat_completion.choices[0].message.content}
        )

    return chat_completion.choices[0].message.content


def init_text_model():
    print("[DEBUG] Initializing text model...")
    prompt = """Prompt:

Break down the following text into smaller and smaller sections while preserving meaning. Each section should be paired with an AI-generated image prompt that visually represents its content. Ensure that:

Each section retains logical coherence and contributes to the full understanding of the original passage.
The sum of all excerpts should contain the original text when combined. All the excerpts should be given expressive and given as ssml formatted input that can be given to a TTS engine.
Each excerpt is paired with a corresponding image prompt that describes what an AI should generate to visualize the excerpt’s meaning.
Output the result in JSON format with the following structure:

json
{
  "breakdown": [
    {
      "excerpt": "[Excerpt from the passage given as ssml formatted input and is very expressive]",
      "image_prompt": "[Detailed AI image prompt describing a visual representation of the excerpt]",
      "raw_excerpt": "[Raw text excerpt]"
    },
    {
      "excerpt": "[Excerpt from the passage given as ssml formatted input and is very expressive]",
      "image_prompt": "[AI image description]",
      "raw_excerpt": "[Raw text excerpt]"
    }
    // Continue this format until the entire passage is broken down
  ]
}
Ensure that the AI image prompts are descriptive and suitable for text-to-image models."""
    reply = get_text_reply(prompt, True)
    print("[DEBUG] Text model initialized successfully")


def convert_output_to_dict(output):
    print("[DEBUG] Converting output to dictionary...")
    curly_1 = output.find("{")
    curly_2 = output.rfind("}")
    output: str = output[curly_1 + 1 : curly_2]
    output = "{" + output + "}"
    try:
        output_dict = json.loads(output)
        print("[DEBUG] Successfully converted output to dictionary")
    except:
        print("[DEBUG] ERROR: Failed to convert output to dictionary")
        print("[DEBUG] Trying Manual Conversion...")

        output_dict = {"breakdown": []}
        splits = output.split('"')
        for i, split in enumerate(splits):
            if split == "excerpt":
                output_dict["breakdown"].append({"excerpt": splits[i + 2]})
            elif split == "image_prompt" or split == "image.prompt":
                output_dict["breakdown"][-1]["image_prompt"] = splits[i + 2]
            elif split == "raw_excerpt":
                output_dict["breakdown"][-1]["raw_excerpt"] = splits[i + 2]

    return output_dict


def convert_output_to_image(output_dict):
    prompts = []
    for item in output_dict["breakdown"]:
        try:
            prompts.append(item["image_prompt"])
        except:
            print("[DEBUG] ERROR: Failed to convert output to image prompts")
            print(item)
            exit(1)
    return prompts


def convert_output_to_raws(output_dict):
    prompts = []
    for item in output_dict["breakdown"]:
        prompts.append(item["raw_excerpt"])
    return prompts


def convert_output_to_post(output_dict):
    post = []
    for item in output_dict["breakdown"]:
        post.append(item["excerpt"])
    return post


def create_text_image(text, width, height, y_position):
    # Create a transparent image for text
    text_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_image)

    # Load font with a proper path
    font = ImageFont.truetype("impact.ttf", 124)  # Increased size to 124 and added bold

    # Handle empty text case
    if not text.strip():
        return text_image

    # Wrap text using textwrap
    wrapped_text = textwrap.fill(text, width=10)  # Adjusted width for larger font
    lines = wrapped_text.split("\n")

    # Compute total text height
    text_height = sum(font.getbbox(line)[3] for line in lines) + (len(lines) - 1) * 20  # Increased line spacing

    # Center text block vertically within the available space
    starting_y = y_position + (height - text_height) // 2
    y = starting_y

    # Draw text
    for line in lines:
        w, h = font.getbbox(line)[2:]
        x = (width - w) // 2  # Center text horizontally
        draw.text((x, y), line, font=font, fill=text_color)  # Using text_color variable
        y += h + 20  # Increased spacing

    return text_image

# New helper: Returns the current word based on time
def get_current_word(full_text, t, total_duration):
    words = full_text.split()
    if not words:
        return ""
    word_duration = total_duration / len(words)
    index = int(t // word_duration)
    if index >= len(words):
        index = len(words) - 1
    return words[index]

# New helper: Creates a dynamic text clip that updates the displayed word based on time
def create_dynamic_text_clip(full_text, duration, width, height, font_path="impact.ttf", font_size=124):
    words = full_text.split()
    word_duration = duration / len(words) if words else duration
    transition_duration = 0.2  # duration of pop-out effect in seconds
    def make_frame(t):
        current_word = get_current_word(full_text, t, duration)
        current_index = int(t // word_duration)
        dt = t - current_index * word_duration
        # Compute pop-out scale: from 0.8 to 1.0 over transition_duration
        if dt < transition_duration:
            scale = 0.8 + 0.2 * (dt / transition_duration)
        else:
            scale = 1.0
        print(f"[DEBUG] Dynamic text clip: time={t:.2f}s, word='{current_word}', scale={scale:.2f}")

        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        effective_font_size = max(1, int(font_size * scale))
        font = ImageFont.truetype(font_path, effective_font_size)

        bbox = draw.textbbox((0, 0), current_word, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        x = (width - w) // 2
        y = (height - h) // 2
        draw.text((x, y), current_word, font=font, fill=text_color)
        return np.array(img)
    return VideoClip(make_frame, duration=duration)


if __name__ == "__main__":
    print("[DEBUG] Starting script execution...")

    init_text_model()

    print("[DEBUG] Fetching and breaking down Reddit post...")
    # Random
    # broken_down_post = reddit.get_post_and_breakdown(5)

    # specifc
    broken_down_post = reddit.get_post_and_breakdown(
        6,
        "https://www.reddit.com/r/ApplyingToCollege/comments/1intpmi/duke_alumni_interview_does_it_affect_admissions/",
    )

    print("[DEBUG] Processing post content...")
    print(broken_down_post["post"])
    breakdown = broken_down_post["breakdown"]

    post_subsections = []
    image_prompts = []
    text_prompts = []

    print("[DEBUG] Processing post breakdowns...")
    for bp in breakdown:
        print(f"[DEBUG] Processing breakdown section: {bp[:50]}...")
        reply = get_text_reply(bp)
        data = convert_output_to_dict(reply)
        prompts = convert_output_to_image(data)
        post_subsections.extend(convert_output_to_post(data))
        image_prompts.extend(prompts)
        try:
            text_prompts.extend(convert_output_to_raws(data))
        except:
            print("[DEBUG] ERROR: Failed to convert output to raws")
            print("data")
            text_prompts.extend([""] * len(prompts))

    print("[DEBUG] Generating images...")
    background.get_bing_images(image_prompts)
    # nested_prompts = [image_prompts[i:i + 5] for i in range(0, len(image_prompts), 5)]
    # for i, _prompts in enumerate(image_prompts):
    #     print(f"[DEBUG] Generating image batch {i+1}/{len(image_prompts)}")
    #     background.get_images(_prompts)

    audio_clips = []
    video_clips = []

    print("[DEBUG] Creating audio and video clips...")
    for idx, subsection in enumerate(post_subsections):
        print(
            f"[DEBUG] Processing clip -> Making audio -> {idx+1}/{len(post_subsections)}"
        )
        audio_path = f"reddit2image/audio/audio_{idx}.mp3"
        tts.tts_to_file(
            text=subsection, file_path=audio_path, speed=random.uniform(1.4, 1.8)
        )

        audio_clip = AudioFileClip(audio_path)
        audio_clips.append(audio_clip)

        print(
            f"[DEBUG] Processing clip -> Making Image -> {idx+1}/{len(post_subsections)}"
        )

        # Create video clip with image and audio
        image_path = f"background/images/{idx}.png"

        # Open and process the image
        img = Image.open(image_path)

        # Create background (blurred version)
        background = img.copy()
        background = background.filter(ImageFilter.GaussianBlur(radius=30))

        # Calculate dimensions for 9:16 ratio
        target_width = 1080  # Standard vertical video width
        target_height = 1920  # 16:9 vertical ratio

        # Resize background to fill the frame
        background = background.resize((target_width, target_height), Image.LANCZOS)
        background = background.convert("RGBA")
        background.putalpha(128)  # 50% opacity

        # Resize original image to fit in center
        img_ratio = min(
            target_width * 0.8 / img.width, target_height * 0.6 / img.height
        )
        new_size = (int(img.width * img_ratio), int(img.height * img_ratio))
        img = img.resize(new_size, Image.LANCZOS)

        # Create new blank image with desired dimensions
        combined = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 255))

        # Paste background
        combined.paste(background, (0, 0))

        # Paste original image in center
        img_pos = (
            (target_width - new_size[0]) // 2,
            (target_height - new_size[1]) // 2,
        )
        combined.paste(img, img_pos)

        # Instead of static text, create a dynamic text clip showing one word at a time
        text_clip_height = 200  # Height for dynamic text area
        text_y_position = img_pos[1] + new_size[1] + 50
        base_clip = ImageClip(np.array(combined)).with_duration(audio_clip.duration)
        dynamic_text_clip = create_dynamic_text_clip(
            text_prompts[idx], audio_clip.duration, target_width, text_clip_height
        ).with_position(("center", text_y_position))

        final_clip = CompositeVideoClip([base_clip, dynamic_text_clip]).with_duration(audio_clip.duration)
        # Adding audio to the final composite
        final_clip = final_clip.with_audio(audio_clip)
        video_clips.append(final_clip)

    print("[DEBUG] Concatenating final video...")
    final_clip = concatenate_videoclips(video_clips)
    final_clip.write_videofile("reddit2image/output/output_video.mp4", fps=24)

    print("[DEBUG] Cleaning up temporary files...")
    for idx in range(len(post_subsections)):
        os.remove(f"reddit2image/audio/audio_{idx}.mp3")
        os.remove(f"background/images/{idx}.png")
        # os.remove(f"background/images/temp_{idx}.png")

    print("[DEBUG] Script execution completed successfully")
