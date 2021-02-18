import pyaudio
import numpy as np
import librosa
from os import write
import os
import scipy.signal as signal
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import threading 
import time
import soundfile

os.remove('./output.txt')


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
CHUNK = 1024

def pred(transcriptor, audio, start):
    number_of_samples = round(x.shape[0] * 16000.0 / RATE)
    X = signal.resample(x, number_of_samples)
    X = X / 20
    stft = librosa.core.stft(X, n_fft=1024, hop_length=256, window='hann', center=True)
    freqs = librosa.fft_frequencies(sr=16000, n_fft=1024)
    for i in range(len(stft)):
        if freqs[i] < 150:
            stft[i] = stft[i]/8
    audio = librosa.istft(stft)
    try:
        result = transcriptor.transcribe(audio, start=start)
    except:
        result = []
    print('thread')
    with open('output.txt', 'a') as file:  
        for note in result:
            file.write(f'{note[1]}\t{note[0]}\n')
    

start = 0

transcriptor = PianoTranscription(device='cpu')    # 'cuda' | 'cpu'

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)


frames = []
frame_1_sec = []
for i in range(3, 0, -1):
    time.sleep(1)
    print(i)
print('start')
for i in range(30):
    for i in range(0, int(RATE / CHUNK * 1)):
        data = stream.read(CHUNK)
        frames.append(data)
        frame_1_sec.append(data)
    data_sec = b''.join(frame_1_sec)
    frame_1_sec = []
    audio_as_np_int16 = np.frombuffer(data_sec, dtype=np.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
    max_int16 = 2**15
    x = audio_as_np_float32 / max_int16
    threading.Thread(target=pred, args=(transcriptor, x, start)).start()
    start += 1
print('end')
