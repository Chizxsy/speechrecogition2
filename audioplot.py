from audiobuf import AudioBuffer
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib
matplotlib.use('TkAgg')


if __name__ == "__main__":
    audio_buffer = AudioBuffer(chunks=16)
    audio_buffer.start()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(-25000, 25000) 
    amp = 4000 # you might need to adjust this parameter
    line, = ax.plot(amp * np.random.random(len(audio_buffer)) - amp/2)
    def animate(i):
        data = audio_buffer()
        line.set_ydata(data)
        return (line,)
    
    anim = animation.FuncAnimation(fig, animate, interval=1, blit=True)
    plt.show()