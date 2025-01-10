import os
import pretty_midi
import numpy as np
import tensorflow as tf
import pandas as pd

# Load the pre-trained model
MODEL_PATH = 'model/my_trained_model.h5'
loaded_model = tf.keras.models.load_model(MODEL_PATH)

# List of instruments for dropdown
INSTRUMENTS = [
    "Acoustic Grand Piano", "Electric Guitar (jazz)", "Violin", "Flute", "Cello",
    "Acoustic Guitar (steel)", "Trumpet", "Saxophone", "Trombone"
]

# Function to process MIDI file and prepare model input
def midi_file_to_input_notes(midi_file: str, seq_length: int) -> np.ndarray:
    pm = pretty_midi.PrettyMIDI(midi_file)
    notes = []

    for instrument in pm.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes.append({
                    'pitch': note.pitch, 
                    'start': note.start, 
                    'end': note.end, 
                    'duration': note.end - note.start
                })

    notes_df = pd.DataFrame(notes).sort_values(by='start')
    notes_df['step'] = notes_df['start'].diff().fillna(0)
    input_notes = notes_df[['pitch', 'step', 'duration']].head(seq_length).to_numpy()

    if len(input_notes) < seq_length:
        padding = np.zeros((seq_length - len(input_notes), 3))
        input_notes = np.vstack([padding, input_notes])

    return input_notes

# Placeholder for `predict_next_note` function
def predict_next_note(input_notes, model, temperature):
    # Replace this with actual prediction code
    return np.random.randint(60, 72), np.random.rand(), np.random.rand()

# Function to generate notes
def generate_notes(input_notes: np.ndarray, model: tf.keras.Model, temperature: float, num_predictions: int) -> pd.DataFrame:
    generated_notes = []
    prev_start = 0

    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration

        generated_notes.append({
            'pitch': pitch, 
            'step': step, 
            'duration': duration, 
            'start': start, 
            'end': end
        })
        input_note = np.array([[pitch, step, duration]])
        input_notes = np.append(input_notes[1:], input_note, axis=0)
        prev_start = start

    return pd.DataFrame(generated_notes)

# Convert generated notes to a MIDI file
def notes_to_midi(generated_notes_df: pd.DataFrame, out_file: str, instrument_name: str) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

    for _, note_data in generated_notes_df.iterrows():
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(note_data['pitch']),
            start=note_data['start'],
            end=note_data['end']
        )
        instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm
