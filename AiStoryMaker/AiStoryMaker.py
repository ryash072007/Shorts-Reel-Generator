import textwrap
from groq import Groq
import json, os, sys, random, time
from moviepy import *
from TTS.api import TTS
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np

sys.path.append("background")
from background import get_fast_images

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

tts = TTS(model_name="tts_models/en/jenny/jenny", progress_bar=True, gpu=False)
text_color = (255, 255, 255, 255)  # White text for better visibility

def generate_story(emotion, additional_prompt=""):
    print(f"[DEBUG] Generating story with emotion: {emotion}")
    if additional_prompt:
        print(f"[DEBUG] Additional prompt: {additional_prompt}")
    prompt = f"""Create a compelling short story (300-500 words) that primarily evokes the emotion of {emotion}. 
    Additional requirements: {additional_prompt}
    
    The story should be vivid, engaging, and suitable for visual storytelling.
    Focus on descriptive language and emotional depth.
    
    Return the story in a simple text format."""
    
    result = get_text_reply(prompt)
    print(f"[DEBUG] Story generated successfully ({len(result)} characters)")
    return result

def break_down_story(story):
    print("[DEBUG] Breaking down story into segments...")
    prompt = f"""Break down this story into 5-8 segments. For each segment:
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
    return convert_output_to_dict(response)

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
        return {"segments": []}

def create_dynamic_text_clip(text, duration, width, height, font_size=80):
    print(f"[DEBUG] Creating dynamic text clip (duration: {duration}s)")
    words = text.split()
    word_duration = duration / len(words) if words else duration
    
    def make_frame(t):
        current_index = int(t // word_duration)
        if current_index % 10 == 0:  # Print every 10th frame to avoid spam
            print(f"[DEBUG] Rendering frame at {t:.2f}s (word {current_index})")
        current_word = words[min(current_index, len(words)-1)]
        
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("impact.ttf", font_size)
        
        bbox = draw.textbbox((0, 0), current_word, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (width - w) // 2
        y = (height - h) // 2
        
        # Add text shadow for better visibility
        shadow_offset = 3
        draw.text((x+shadow_offset, y+shadow_offset), current_word, font=font, fill=(0, 0, 0, 255))
        draw.text((x, y), current_word, font=font, fill=text_color)
        
        return np.array(img)
    
    return VideoClip(make_frame, duration=duration)

def create_video(segments):
    print(f"[DEBUG] Creating video with {len(segments)} segments")
    video_clips = []
    
    for idx, segment in enumerate(segments):
        print(f"[DEBUG] Processing segment {idx+1}/{len(segments)}")
        
        print(f"[DEBUG] Generating audio for segment {idx+1}")
        audio_path = f"AiStoryMaker/audio/segment_{idx}.mp3"
        tts.tts_to_file(text=segment["narration"], file_path=audio_path, speed=1.2)
        print(f"[DEBUG] Audio saved to {audio_path}")
        audio_clip = AudioFileClip(audio_path)
        
        print(f"[DEBUG] Generating image for segment {idx+1}")
        get_fast_images([segment["image_prompt"]], prefix="", start=idx)
        print(f"[DEBUG] Image generated successfully")
        image_path = f"background/images/{idx}.png"  # Only one image per segment
        
        # Create video frame
        img = Image.open(image_path)
        target_width, target_height = 1080, 1920  # 9:16 ratio
        
        # Resize image to fill frame
        img = img.resize((target_width, target_height), Image.LANCZOS)
        
        # Add slight blur to background
        img = img.filter(ImageFilter.GaussianBlur(radius=3))
        
        # Create base clip with image
        base_clip = ImageClip(np.array(img)).with_duration(audio_clip.duration)
        
        # Create dynamic text clip
        text_clip = create_dynamic_text_clip(
            segment["raw_text"],
            audio_clip.duration,
            target_width,
            200  # Height for text area
        ).with_position(("center", target_height//2))
        
        # Combine clips
        final_clip = CompositeVideoClip([base_clip, text_clip])
        final_clip = final_clip.with_audio(audio_clip)
        video_clips.append(final_clip)
        print(f"[DEBUG] Segment {idx+1} processing complete")
        
    print("[DEBUG] Concatenating all segments...")
    final_video = concatenate_videoclips(video_clips)
    print("[DEBUG] Video concatenation complete")
    return final_video

def main(emotion, additional_prompt=""):
    print("[DEBUG] Starting AiStoryMaker...")
    print(f"[DEBUG] Creating directories...")
    # Create necessary directories
    os.makedirs("AiStoryMaker/audio", exist_ok=True)
    os.makedirs("AiStoryMaker/output", exist_ok=True)
    
    print("[DEBUG] Story generation started")
    print("Generating story...")
    story = generate_story(emotion, additional_prompt)
    
    print("[DEBUG] Story breakdown started")
    print("Breaking down story into segments...")
    story_data = break_down_story(story)
    print(f"[DEBUG] Story broken down into {len(story_data['segments'])} segments")
    
    print("[DEBUG] Video creation started")
    print("Creating video...")
    final_video = create_video(story_data["segments"])
    
    print("[DEBUG] Saving final video...")
    print("Saving video...")
    output_path = "AiStoryMaker/output/story.mp4"
    final_video.write_videofile(output_path, fps=24)
    
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

if __name__ == "__main__":
    print("[DEBUG] AiStoryMaker script started")
    emotion = input("Enter the primary emotion for the story: ")
    additional_prompt = input("Enter any additional requirements (or press Enter to skip): ")
    main(emotion, additional_prompt)
