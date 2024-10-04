import torch
from TTS.api import TTS

# Get the device, GUP if available
device = "cuda" if torch.cuda.is_available() else "cpu"

#loads an available model, jenny's voice
tts = TTS("tts_models/en/jenny/jenny").to(device)

#"Sample text" needs to be replaced by the bot's response, in str format
tts.tts_to_file(text="Sample text", file_path="output.wav")

print("Audio saved as output.wav")