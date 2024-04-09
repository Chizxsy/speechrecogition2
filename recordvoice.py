import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile

CHUNK = 1000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Start recording...")
frames = []
for _ in range(int(RATE / CHUNK)):
    data = stream.read(CHUNK)
    frames.append(np.frombuffer(data, dtype=np.int16))

audio_data = np.hstack(frames)
print("Stopped recording")

def normalize(audio_data):
    max_value = np.max(np.abs(audio_data))
    normalized_data = audio_data / max_value
    return normalized_data

normalized_data = normalize(audio_data)
scipy.io.wavfile.write("output.wav", RATE, audio_data)

time = np.arange(0, len(normalized_data)) / RATE

plt.figure(figsize=(10, 4))
plt.plot(normalized_data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Waveform')
plt.grid(True)
plt.show()

stream.stop_stream()
stream.close()
p.terminate()
