from groq import Groq
import json, os, sys
from moviepy import *
from moviepy import *
from TTS.api import TTS
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np
import random

sys.path.append("background")
sys.path.append("reddit")

import background, reddit

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

tts = TTS(model_name="tts_models/en/jenny/jenny", progress_bar=True, gpu=False)

history = []


def get_text_reply(prompt, add_to_memory=False):
    global messages

    chat_completion = client.chat.completions.create(
        messages=history
        + [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gemma2-9b-it",
    )

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
    output = output[curly_1 + 1 : curly_2]
    output = "{" + output + "}"
    try:
        output_dict = json.loads(output)
        print("[DEBUG] Successfully converted output to dictionary")
    except:
        print("[DEBUG] ERROR: Failed to convert output to dictionary")
        print(output)
    return output_dict


def convert_output_to_image(output_dict):
    prompts = []
    for item in output_dict["breakdown"]:
        prompts.append(item["image_prompt"])
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
    text_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_image)
    
    # Load a font (you'll need to specify the correct path to a font file)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Wrap text to fit width
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        w, h = draw.textsize(' '.join(current_line), font=font)
        if w > width * 0.8:  # 80% of width
            current_line.pop()
            lines.append(' '.join(current_line))
            current_line = [word]
    lines.append(' '.join(current_line))
    
    # Draw text
    text_height = sum(draw.textsize(line, font=font)[1] for line in lines)
    y = y_position
    
    for line in lines:
        w, h = draw.textsize(line, font=font)
        x = (width - w) // 2  # Center text horizontally
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += h + 10  # Add some spacing between lines
    
    return text_image


if __name__ == "__main__":
    print("[DEBUG] Starting script execution...")

    init_text_model()

    print("[DEBUG] Fetching and breaking down Reddit post...")
    broken_down_post = reddit.get_post_and_breakdown(5)

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
        text_prompts.extend(convert_output_to_raws(data))

    print("[DEBUG] Generating images...")
    nested_prompts = [image_prompts[i:i + 5] for i in range(0, len(image_prompts), 5)]
    for i, _prompts in enumerate(nested_prompts):
        print(f"[DEBUG] Generating image batch {i+1}/{len(nested_prompts)}")
        background.get_images(_prompts)

    audio_clips = []
    video_clips = []

    print("[DEBUG] Creating audio and video clips...")
    for idx, subsection in enumerate(post_subsections):
        print(f"[DEBUG] Processing clip -> Making audio -> {idx+1}/{len(post_subsections)}")
        audio_path = f"reddit2image/audio/audio_{idx}.mp3"
        tts.tts_to_file(text=subsection, file_path=audio_path, speed=random.uniform(1.2, 1.8))

        audio_clip = AudioFileClip(audio_path)
        audio_clips.append(audio_clip)

        print(f"[DEBUG] Processing clip -> Making Image -> {idx+1}/{len(post_subsections)}")

        # Create video clip with image and audio
        image_path = f"background/images/{idx}.jpeg"
        
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
        background = background.convert('RGBA')
        background.putalpha(128)  # 50% opacity
        
        # Resize original image to fit in center
        img_ratio = min(target_width * 0.8 / img.width, target_height * 0.6 / img.height)
        new_size = (int(img.width * img_ratio), int(img.height * img_ratio))
        img = img.resize(new_size, Image.LANCZOS)
        
        # Create new blank image with desired dimensions
        combined = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 255))
        
        # Paste background
        combined.paste(background, (0, 0))
        
        # Paste original image in center
        img_pos = ((target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2)
        combined.paste(img, img_pos)

        # Create and paste text
        text_y_position = img_pos[1] + new_size[1] + 50  # 50 pixels below the image
        text_image = create_text_image(text_prompts[idx], target_width, target_height - text_y_position, 0)
        combined.paste(text_image, (0, text_y_position), text_image)
        
        # Save temporary combined image
        temp_path = f"background/images/temp_{idx}.png"
        combined.save(temp_path)

        # Create video clip with combined image
        image_clip = (
            ImageClip(temp_path)
            .with_duration(audio_clip.duration)
            .with_audio(audio_clip)
        )
        video_clips.append(image_clip)

    print("[DEBUG] Concatenating final video...")
    final_clip = concatenate_videoclips(video_clips)
    final_clip.write_videofile("reddit2image/output/output_video.mp4", fps=24)

    print("[DEBUG] Cleaning up temporary files...")
    for idx in range(len(post_subsections)):
        os.remove(f"reddit2image/audio/audio_{idx}.mp3")
        os.remove(f"background/images/{idx}.jpeg")
        os.remove(f"background/images/temp_{idx}.png")
    
    print("[DEBUG] Script execution completed successfully")
