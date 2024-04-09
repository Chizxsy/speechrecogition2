import numpy as np
import pyaudio
import tensorflow as tf


CHUNK = 2000 #number of frames per buffer
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Initialize PyAudio
p = pyaudio.PyAudio()

commands =['down', 'up']



def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def preprocess_audio(waveform):
    waveform = waveform/32768
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

    spectrogram = get_spectrogram(waveform)
    spectrogram = tf.expand_dims(spectrogram, 0)

    return spectrogram

def rec_audio():
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
   
    
    frames = []
    print("rec...(1s)")
    d = 1
    for i in range(0,int(RATE/(CHUNK*d))):
        data = stream.read(CHUNK)
        frames.append(data)

    print(len(frames))
    stream.stop_stream()
    stream.close()
    waveform = np.frombuffer(b''.join(frames), dtype=np.int16)
    #print(waveform)

    return waveform

def close():
    p.terminate()



interpreter = tf.lite.Interpreter(model_path="voicemodel.tflite")
interpreter.allocate_tensors()

def inference():
    audio = rec_audio()
    input_data = preprocess_audio(audio)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
  
    # Process the output data and perform actions based on predictions
    predicted_class = np.argmax(output_data)
    confidence = output_data[0][predicted_class]
    

    # Perform actions based on the predicted class and confidence
    if confidence > 2.5:  # test starting with 0.5
  
        predicted_label = commands[predicted_class]
        print(f"Predicted Command: {predicted_label} (Confidence: {confidence:.2f})")
   
        
    else:
        print("No clear command detected")
     

if __name__=="__main__":
    try:
        while True:
            inference()
    except KeyboardInterrupt:
        pass
            
