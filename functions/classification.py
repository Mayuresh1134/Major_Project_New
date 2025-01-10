from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import librosa
import tensorflow as tf
import os

# Load the trained model
MODEL_PATH = 'model1.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Define the genre labels (assuming 10 genres from GTZAN dataset)
GENRE_LABELS = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Path to save uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Preprocess the uploaded audio file by extracting MFCCs
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, duration=30)  # Load 30 seconds of audio
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract 13 MFCC coefficients
    mfccs_scaled = np.mean(mfccs.T, axis=0)  # Mean normalization
    mfccs_scaled = mfccs_scaled.reshape(1, -1)  # Reshape to match the model input
    return mfccs_scaled

# Predict genre from audio file
def predict_genre(model, file_path):
    mfccs_scaled = preprocess_audio(file_path)
    prediction = model.predict(mfccs_scaled)
    predicted_genre = GENRE_LABELS[np.argmax(prediction)]
    return predicted_genre

