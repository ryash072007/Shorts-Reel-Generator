
import os
import uuid
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)


def text_to_speech_file(text: str) -> str:
    # Calling the text_to_speech conversion API with detailed parameters
    response = client.text_to_speech.convert(
        voice_id="I4MFG1v2Ntx1WlZeBovR", # Adam pre-made voice
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_flash_v2_5", # use the turbo model for low latency
        # Optional voice settings that allow you to customize the output
        voice_settings=VoiceSettings(
            stability=0.35,
            similarity_boost=0.55,
            style=0.0,
            use_speaker_boost=True,
            speed=1.15,
        ),
    )

    # uncomment the line below to play the audio back
    # play(response)

    # Generating a unique file name for the output MP3 file
    save_file_path = f"{uuid.uuid4()}.mp3"

    # Writing the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")

    # Return the path of the saved audio file
    return save_file_path

text_to_speech_file("""<speak>
  <prosody rate="50%" pitch="-10%">Helloooo... </prosody> 
  <break time="1s"/>
  <prosody rate="200%" pitch="+10%">THIS IS A SUPER FAST PART!</prosody>  
  <break time="500ms"/>
  Now, let's add some <emphasis level="strong">EMPHASIS!</emphasis>  
  <break time="1s"/>
  And a dramatic... <break time="2s"/> pause.
  <break time="500ms"/>
  <prosody volume="x-loud">DID THIS WORK?!</prosody>
</speak>
""")
