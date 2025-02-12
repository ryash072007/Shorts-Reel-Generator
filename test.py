import requests
from bs4 import BeautifulSoup

# # # # from pyt2s.services import stream_elements

# # # # # Custom Voice
# # # # data = stream_elements.requestTTS('Lorem Ipsum is simply dummy text.', stream_elements.Voice.Nicole.value)

# # # # with open('output.mp3', '+wb') as file:
# # # #     file.write(data)


# # # from TTS.api import TTS

# # # # tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False)
# # # # tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=True, gpu=False)
# # # tts = TTS(model_name="tts_models/en/jenny/jenny", progress_bar=True, gpu=False)
# # # tts.tts_to_file(text="""Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
# # #       But I also have other interests such as playing tic tac toe.""", file_path="output.mp3", speed=1.5)

# # # # from bark import SAMPLE_RATE, generate_audio, preload_models
# # # # from scipy.io.wavfile import write as write_wav


# # # # # download and load all models
# # # # preload_models()

# # # # # generate audio from text
# # # # text_prompt = """
# # # #      Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
# # # #      But I also have other interests such as playing tic tac toe.
# # # # """

# # # # speech_array1 = generate_audio(text_prompt, history_prompt="en_speaker_1")

# # # # # audio_array = generate_audio(text_prompt)

# # # # # save audio to disk
# # # # write_wav("bark_generation.wav", SAMPLE_RATE, speech_array1)

# # from TTS.api import TTS

# # # Initialize the TTS engine
# # tts = TTS(model_name="tts_models/en/jenny/jenny")  # Replace with your model name

# # # Define your SSML input
# # ssml_input = """
# # <speak>
# #     <voice name="en-US-Standard-C">
# #         <prosody rate="slow" pitch="+2st">Hello, welcome to Coqui TTS.</prosody>
# #     </voice>
# #     <break time="500ms"/>
# #     <prosody rate="medium">This is an example of using SSML with Coqui TTS.</prosody>
# # </speak>
# # """

# # tts.tts_to_file(text=ssml_input, file_path="output.mp3")

# # # Generate speech from SSML
# # # tts.speak(ssml_input)

# output = """
# {
#   "breakdown": [
#     {
#       "excerpt": "<say-as interpret-as='interjection'> Wow! </say-as> This is a crazy, wild story, and I definitely don't recommend anyone trying it, for obvious reasons. But, here it goes.  ",
#       "image_prompt": "A person with a shocked and surprised expression, hands up in the air as if telling a unbelievable story.",
#       "raw_excerpt": "This is a crazy, wild story, and I definitely don't recommend anyone trying it, for obvious reasons. But, here it goes."     
#     },
#     {
#       "excerpt": "<say-as interpret-as='character'> I </say-as>’m both a bird hunter and recreational sports shooter (my cousin taught me a lot since I was 8) and this story happened a few months after turning 21.",
#       "image_prompt": "A person holding a rifle, wearing camouflage gear, against a backdrop of forest.",
#       "raw_excerpt": "I’m both a bird hunter and recreational sports shooter (my cousin taught me a lot since I was 8) and this story happened a few months after turning 21."
#     },
#     {
#       "excerpt": "<say-as interpret-as='character'> My </say-as> parents let's just say were very against guns at the time and I had just gotten my first apartment so I was extremely excited to own my first gun without getting shit from my parents lol.  ",
#       "image_prompt": "A young adult standing outside a modest apartment building, holding a key in hand and smiling triumphantly.",
#       "raw_excerpt": "My parents let's just say were very against guns at the time and I had just gotten my first apartment so I was extremely excited to own my first gun without getting shit from my parents lol."
#     },
#     {
#       "excerpt": "<say-as interpret-as='character'> I </say-as> went to a store called sportsman warehouse and bought a 9mm Ruger security 9.",    
#       "image_prompt": "A person  inside a large sporting goods store, examining a 9mm Ruger security 9 at display case.",
#       "raw_excerpt": "I went to a store called sportsman warehouse and bought a 9mm Ruger security 9."
#     },
#     {
#       "excerpt": "<say-as interpret-as='character'> I </say-as> love Ruger's in particular when it comes to rifles and pistols so that was my choice",
#       "image_prompt": "A close-up image of a hand resting on a Ruger rifle, with a look of admiration in the person's eyes.",
#       "raw_excerpt": "I love Ruger's in particular when it comes to rifles and pistols so that was my choice"
#     },
#     {
#       "excerpt": "<say-as interpret-as='character'> So </say-as> when I got to my car (in a public parking lot with cops doing a drug bust pullover nearby) let's just say I did a lot of shit that you shouldn't absloutely do and then I unloaded my gun after loading it, then I pointed directly right under my radio near the engine and pulled the trigger.",
#       "image_prompt": "A parked car in a crowded public parking lot.  A figure is leaning into the car, the side of the car is illuminated with a bluish light.  ",
#       "raw_excerpt": "So when I got to my car (in a public parking lot with cops doing a drug bust pullover nearby) let's just say I did a lot of shit that you shouldn't absloutely do and then I unloaded my gun after loading it, then I pointed directly right under my radio near the engine and pulled the trigger."
#     },
#     {
#       "excerpt": "<say-as interpret-as='interjection'> Wow! </say-as> Let's just say I had no idea a bullet was still in the chamber and next thing ya know my ears were ringing like a motha fucka and I was scared shitless.  ",
#       "image_prompt": "A silhouette of a person inside a car,  arms up and head tilted back in shock",
#       "raw_excerpt": "Let's just say I had no idea a bullet was still in the chamber and next thing ya know my ears were ringing like a motha fucka and I was scared shitless."
#     },
#     {
#       "excerpt": "<say-as interpret-as='character'> Like </say-as> I said earlier there were cops nearby so immediately got out of there and brought the car back to my house were it had to be towed. I decided to own it up to my parents and help them pay for the stupid shit I caused overtime", 
#       "image_prompt": "Empty car parked in a lot, being investigated by two police officers wearing reflective vests.",
#       "raw_excerpt": "Like I said earlier there were cops nearby so immediately got out of there and brought the car back to my house were it had to be towed. I decided to own it up to my parents and help them pay for the stupid shit I caused overtime"
#     },
#     {
#       "excerpt": "<say-as interpret-as='interjection'> Thankfully, </say-as> nobody was hurt and I wasn't arrested for reckless endangerment, but lesson for all y'all to learn here, be responsible and use every common sense you have when using guns.  And absloutely do not be messing around with guns like there toys especially in public.",
#       "image_prompt": "A group of people gathered around a table, looking concerned, while a single person solemnly holds a firearm in their hands,

#       "raw_excerpt": "Thankfully, nobody was hurt and I wasn't arrested for reckless endangerment, but lesson for all y'all to learn here, be responsible and use every common sense you have when using guns.  And absloutely do not be messing around with guns like there toys especially in public."
#     }
#   ]
# }
# """


# output_dict = {"breakdown": []}
# splits = output.split('"')
# for i, split in enumerate(splits):
#     if split == "excerpt":
#         output_dict["breakdown"].append({"excerpt": splits[i + 2]})
#     elif split == "image_prompt":
#         output_dict["breakdown"][-1]["image_prompt"] = splits[i + 2]
#     elif split == "raw_excerpt":
#         output_dict["breakdown"][-1]["raw_excerpt"] = splits[i + 2]

# print(output_dict)

# Function to fetch the content of a Reddit post
import requests

# URL of the Reddit post
url = "https://www.reddit.com/r/IntltoUSA/comments/1inrn3l/the_system_is_against_you_international_babies/.json"

# Set a user-agent to avoid being blocked
headers = {"User-Agent": "Mozilla/5.0"}

# Fetch the JSON data
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    
    # Get the title
    title = data[0]['data']['children'][0]['data']['title']
    
    # Get the post content (selftext)
    post_content = data[0]['data']['children'][0]['data']['selftext']
    
    # Print the results
    print("Title:", title)
    print("\nContent:", post_content if post_content else "No content available")

else:
    print(f"Failed to fetch the post. Status code: {response.status_code}")
