from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from IPython.display import Audio
from pynput import keyboard
from pynput.keyboard import Key
from pynput.keyboard import Controller as KeyboardController
import subprocess
import shlex
import os
import sounddevice as sd
import numpy as np
import queue
import torch

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# Keyboard controller instance to emulate keystrokes
keyboard_controller = KeyboardController()

# Set the audio recording parameters
samplerate = 16000  # Sample rate in Hertz
channels = 1  # Number of audio channels (1 for mono, 2 for stereo)
duration = None  # Duration is none because we want to start and stop recording manually

# Queue to hold the recorded audio frames
audio_queue = queue.Queue()

# This flag will control the recording state
is_recording = False

# Audio callback function
def audio_callback(indata, frames, time, status):
    """This is called for each audio block from the microphone."""
    if status:
        print(status)
    audio_queue.put(indata.copy())

# Keyboard listener events
def on_press(key):
    global is_recording
    if key == Key.f8:
        if not is_recording:
            print("Recording started...")
            is_recording = True
            # Start recording
            with sd.InputStream(callback=audio_callback, samplerate=samplerate, channels=channels):
                sd.sleep(1000)  # This is just to keep the recording stream open; adjust as needed
        else:
            print("Recording stopped...")
            is_recording = False
            # Stop recording and process audio
            process_audio()

def process_audio():
    global frame
    frame = torch.tensor([0])
    # Stop the input stream and process the recorded audio
    while not audio_queue.empty():
        frame = torch.cat((frame,torch.tensor(np.hstack(audio_queue.get()))))
        # Here, you would add the frames to a buffer or file
        # For example, let's print the shape of the numpy array
        # print(frame.shape)
        # You would use Whisper here to transcribe the audio
        
    input_features = processor(
        frame,
        sampling_rate=samplerate,
        return_tensors="pt"
    ).input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print("output:",transcription)
    keyboard_controller.type(transcription[0])

# Collect events until released
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
