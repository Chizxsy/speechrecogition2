import pyaudio
import numpy as np
from collections import deque
import threading
from time import sleep


class AudioBuffer:

    RATE = 16000
    CHUNK = 1000
    
    def __init__(self, chunks: int = 16) -> None:
        self.chunks = chunks
        self.stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            )
        self.thread = threading.Thread(target=self._collect_data, daemon=True)
        self.frames = deque(maxlen=self.chunks)

    def __call__(self):
        return np.concatenate(self.frames)
    
    def __len__(self):
        return self.CHUNK * self.chunks
    
    def is_full(self):
        return len(self.frames) == self.chunks
    
    def start(self):
        self.thread.start()
        while not self.is_full(): # wait until the buffer is filled
            sleep(0.1)
        
    def _collect_data(self):
        while True:
            raw_data = self.stream.read(self.CHUNK)
            decoded = np.frombuffer(raw_data, np.int16)
            self.frames.append(decoded)


if __name__ == "__main__":
    audio_buffer = AudioBuffer()
    audio_buffer.start()
    print(audio_buffer())
    print(audio_buffer().shape)