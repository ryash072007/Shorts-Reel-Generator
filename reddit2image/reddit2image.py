from groq import Groq
import json, os, sys
import numpy as np
from moviepy import *
from PIL import Image
from gtts import gTTS
from moviepy import *
from pyht import Client
from TTS.api import TTS
# from pyht.client import TTSOptions

# from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip

sys.path.append("background")
sys.path.append("reddit")

import background, reddit

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

tts = TTS(model_name="tts_models/en/jenny/jenny", progress_bar=True, gpu=False)

# clientV = Client(
#     user_id=os.getenv("PLAY_HT_USER_ID"),
#     api_key=os.getenv("PLAY_HT_API_KEY"),
# )
# options = TTSOptions(voice="s3://voice-cloning-zero-shot/775ae416-49bb-4fb6-bd45-740f205d20a1/jennifersaad/manifest.json")

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
    prompt = """Prompt:

Break down the following text into smaller and smaller sections while preserving meaning. Each section should be paired with an AI-generated image prompt that visually represents its content. Ensure that:

Each section retains logical coherence and contributes to the full understanding of the original passage.
The sum of all excerpts should reconstruct the original text when combined.
Each excerpt is paired with a corresponding image prompt that describes what an AI should generate to visualize the excerpt’s meaning.
Output the result in JSON format with the following structure:

json
{
  "breakdown": [
    {
      "excerpt": "[Excerpt from the passage]",
      "image_prompt": "[Detailed AI image prompt describing a visual representation of the excerpt]"
    },
    {
      "excerpt": "[Next excerpt]",
      "image_prompt": "[AI image description]"
    }
    // Continue this format until the entire passage is broken down
  ]
}
Ensure that the AI image prompts are descriptive and suitable for text-to-image models."""
    reply = get_text_reply(prompt, True)


def convert_output_to_dict(output):
    curly_1 = output.find("{")
    curly_2 = output.rfind("}")
    output = output[curly_1 + 1 : curly_2]
    output = "{" + output + "}"
    try:
        output_dict = json.loads(output)
    except:
        print(output)
    return output_dict


def convert_output_to_image(output_dict):
    prompts = []
    for item in output_dict["breakdown"]:
        prompts.append(item["image_prompt"])
    return prompts


def convert_output_to_post(output_dict):
    post = []
    for item in output_dict["breakdown"]:
        post.append(item["excerpt"])
    return post


# frame_id = 0
# i = 0
# def get_frame(t):
#     global frame_id, i
#     frame = Image.open(f"background/images/{frame_id}.jpeg")
#     i += 1
#     if i % 24 == 0:
#         frame_id += 1
#     return np.array(frame)


if __name__ == "__main__":

    init_text_model()

    broken_down_post = reddit.get_post_and_breakdown(5)

    print(broken_down_post["post"])
    breakdown = broken_down_post["breakdown"]

    post_subsections = []

    i = 0
    total = 0
    for bp in breakdown:
        print(bp)
        reply = get_text_reply(bp)
        data = convert_output_to_dict(reply)
        prompts = convert_output_to_image(data)
        post_subsections.extend(convert_output_to_post(data))
        nested_prompts = [prompts[i:i + 8] for i in range(0, len(prompts), 8)]
        for _prompts in nested_prompts:
            background.get_images(_prompts, "", total)
            total += len(_prompts)

    audio_clips = []
    video_clips = []

    # tts = TTS("tts_models/en/ljspeech/vits")

    for idx, subsection in enumerate(post_subsections):
        # Convert text to speech

        # tts = gTTS(subsection)
        # audio_path = f"audio_{idx}.mp3"
        # tts.save(audio_path)

        # audio_path = f"audio_{idx}.wav"
        # tts.tts_to_file(subsection, audio_path)

        # audio_path = f"audio_{idx}.mp3"
        # with open(audio_path, "wb") as audio_file:
        #     for chunk in clientV.tts(subsection, options, voice_engine = 'PlayDialog-http'):
        #         audio_file.write(chunk)

        audio_path = f"reddit2image/audio/audio_{idx}.mp3"
        tts.tts_to_file(text=subsection, file_path=audio_path, speed=1.5)

        audio_clip = AudioFileClip(audio_path)
        audio_clips.append(audio_clip)

        # Create video clip with image and audio
        image_path = f"background/images/{idx}.jpeg"
        image_clip = (
            ImageClip(image_path)
            .with_duration(audio_clip.duration)
            .with_audio(audio_clip)
        )
        # image_clip = image_clip.set_audio(audio_clip)
        video_clips.append(image_clip)

    # Concatenate all video clips
    final_clip = concatenate_videoclips(video_clips)
    final_clip.write_videofile("reddit2image/output/output_video.mp4", fps=24)

    # Clean up temporary audio files
    for idx in range(len(post_subsections)):
        os.remove(f"reddit2image/audio/audio_{idx}.mp3")
    for idx in range(len(post_subsections)):
        os.remove(f"background/images/{idx}.jpeg")
