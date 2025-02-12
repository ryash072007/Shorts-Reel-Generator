# from pyt2s.services import stream_elements

# # Custom Voice
# data = stream_elements.requestTTS('Lorem Ipsum is simply dummy text.', stream_elements.Voice.Nicole.value)

# with open('output.mp3', '+wb') as file:
#     file.write(data)


from TTS.api import TTS

# tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False)
# tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=True, gpu=False)
tts = TTS(model_name="tts_models/en/jenny/jenny", progress_bar=True, gpu=False)
tts.tts_to_file(text="""Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
      But I also have other interests such as playing tic tac toe.""", file_path="output.mp3", speed=1.5)

# from bark import SAMPLE_RATE, generate_audio, preload_models
# from scipy.io.wavfile import write as write_wav


# # download and load all models
# preload_models()

# # generate audio from text
# text_prompt = """
#      Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
#      But I also have other interests such as playing tic tac toe.
# """

# speech_array1 = generate_audio(text_prompt, history_prompt="en_speaker_1")

# # audio_array = generate_audio(text_prompt)

# # save audio to disk
# write_wav("bark_generation.wav", SAMPLE_RATE, speech_array1)
