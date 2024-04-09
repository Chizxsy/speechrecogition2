from audiobuf import AudioBuffer
import numpy as np
import tensorflow as tf
import time

commands = ['down', 'up']

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

def wave():
    waveform = audio_buffer()
    return waveform

interpreter = tf.lite.Interpreter(model_path="voicemodel.tflite")
interpreter.allocate_tensors()

def inference():
    audio_data = audio_buffer()
    input_data = preprocess_audio(audio_data)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
  
    # Process the output data and perform actions based on predictions
    predicted_class = np.argmax(output_data)
    confidence = output_data[0][predicted_class]
    

    # Perform actions based on the predicted class and confidence
    if confidence > 3:  
  
        predicted_label = commands[predicted_class]
        print(f"Predicted Command: {predicted_label} (Confidence: {confidence:.2f})")
   
        
    else:
        print("No clear command detected")

if __name__ == "__main__":
    audio_buffer = AudioBuffer(chunks=16)
    audio_buffer.start()
    for i in range(10):
        inference()

