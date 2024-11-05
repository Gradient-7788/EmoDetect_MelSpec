import streamlit as st
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model

# Load your pre-trained model
model_path = "emotion_detector.h5"
model = load_model(model_path)

# Define emotion labels
emotion_names = ['Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']

# Streamlit app title
st.title("Real-Time Speech Emotion Detection")

# Set parameters for recording
duration = 5  # seconds
sample_rate = 22050  # Hz (matches librosa's default sample rate)

# Record audio function
def record_audio():
    st.write("Recording for {} seconds...".format(duration))
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    st.write("Recording completed!")
    return audio.flatten()

# Preprocess the audio to extract mel spectrograms
import librosa
import numpy as np

import librosa
import numpy as np

def preprocess_audio(audio, sample_rate=22050, target_shape=(128, 154)):
    # Ensure audio is in the correct format
    audio = np.float32(audio)
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=target_shape[0])
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Adjust the time dimension to match the target shape
    if mel_spec_db.shape[1] > target_shape[1]:  # If too long, truncate
        mel_spec_db = mel_spec_db[:, :target_shape[1]]
    elif mel_spec_db.shape[1] < target_shape[1]:  # If too short, pad with zeros
        padding = target_shape[1] - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, padding)), mode='constant')
    
    # Add channel dimension and repeat to make 3 channels
    mel_spec_db = mel_spec_db[..., np.newaxis]
    mel_spec_db = np.repeat(mel_spec_db, 3, axis=-1)
    
    return mel_spec_db



# Predict emotion function
def predict_emotion(audio):
    # Preprocess audio
    processed_audio = preprocess_audio(audio, sample_rate)
    processed_audio = np.expand_dims(processed_audio, axis=0)  # Add batch dimension
    # Predict with the model
    predictions = model.predict(processed_audio)
    predicted_class = np.argmax(predictions, axis=1)[0]
    emotion = emotion_names[predicted_class]
    return emotion

# Streamlit UI for recording and predicting
if st.button("Record Audio"):
    audio = record_audio()
    # Save audio temporarily to check (optional)
    write("temp_audio.wav", sample_rate, audio)
    
    # Predict the emotion
    emotion = predict_emotion(audio)
    
    # Display result
    st.write("Predicted Emotion:", emotion)
