import subprocess
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, send_from_directory
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from functions.classification import *
from functions.midi_to_wav import *
import uuid
from functions.midi_to_sheet import allowed_file, convert_midi_to_pdf, convert_pdf_to_png, setup_directories
from functions.generation import *
import subprocess


app = Flask(__name__)

setup_directories(app)
app.secret_key = "secret_key"

# Upload folder configuration
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# List of instruments for dropdown
INSTRUMENTS = [
    "Acoustic Grand Piano", "Electric Guitar (jazz)", "Violin", "Flute", "Cello",
    "Acoustic Guitar (steel)", "Trumpet", "Saxophone", "Trombone"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sheet2music')
def s2m():
    return render_template('sheet2music.html', audio_file=None)

@app.route('/upload_midi', methods=['POST'])
def upload_file_midi():
    if 'file' not in request.files or 'instrument' not in request.form:
        return redirect(url_for('s2m'))

    file = request.files['file']
    instrument = request.form['instrument']

    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('s2m'))

    # Call the function from midi_to_wav.py
    output_filename, error = convert_midi_to_wav(file, instrument)
    if error:
        return f"Error: {error}", 500

    # Render the page with the converted audio file
    return render_template('sheet2music.html', audio_file=f"audio/{output_filename}")

@app.route('/music2sheet')
def m2s():
    return render_template('music2sheet.html')

@app.route('/upload_midi_sheet', methods=['POST'])
def upload_midi_sheet():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('index'))
    
    # Save the uploaded MIDI file
    midi_filename = f"{uuid.uuid4().hex}.mid"
    midi_path = os.path.join(app.config['UPLOAD_FOLDER'], midi_filename)
    file.save(midi_path)
    
    # Generate output PDF path
    output_pdf_filename = f"{uuid.uuid4().hex}.pdf"
    output_pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], output_pdf_filename)
    
    # Convert MIDI to PDF
    try:
        convert_midi_to_pdf(midi_path, output_pdf_path)
    except subprocess.CalledProcessError:
        return "Error in converting MIDI to sheet music", 500
    
    # Convert PDF to PNG
    output_png_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{uuid.uuid4().hex}.png")
    try:
        convert_pdf_to_png(output_pdf_path, output_png_path)
    except Exception as e:
        return f"Error converting PDF to PNG: {e}", 500
    
    # Serve the PNG for download
    return send_file(output_png_path, as_attachment=True, download_name="sheet_music.png")

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/tools')
def tool():
    return render_template('tools.html')

@app.route("/generation", methods=["GET", "POST"])
def gen():
    if request.method == "POST":
        midi_file = request.files["midi_file"]
        temperature = float(request.form["temperature"])
        num_predictions = int(request.form["num_predictions"])
        instrument_name = request.form["instrument_name"]


        if midi_file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], midi_file.filename)
            midi_file.save(file_path)

            input_notes = midi_file_to_input_notes(file_path, seq_length=25)
            generated_notes_df = generate_notes(input_notes, loaded_model, temperature, num_predictions)
            output_midi_file = os.path.join(app.config['UPLOAD_FOLDER'], 'generated_music_output.midi')
            notes_to_midi(generated_notes_df, out_file=output_midi_file, instrument_name=instrument_name)

            flash("MIDI generation complete!", "success")
            return send_file(output_midi_file, as_attachment=True)
        
    # Pass `instruments` on both GET and POST methods
    return render_template("generation.html", instruments=INSTRUMENTS)

@app.route('/style_transfer')
def style():
    return render_template('style_transfer.html')

@app.route('/upload_transfer', methods=['POST'])
def upload_files():
    if 'contentAudio' not in request.files or 'styleAudio' not in request.files:
        flash('No file part')
        return redirect(request.url)

    content_audio = request.files['contentAudio']
    style_audio = request.files['styleAudio']

    if content_audio.filename == '' or style_audio.filename == '':
        flash('No selected file')
        return redirect(request.url)

    # Save the files to the upload folder
    content_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], content_audio.filename)
    style_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], style_audio.filename)
    content_audio.save(content_audio_path)
    style_audio.save(style_audio_path)

    print("Running NeuralStyleTransfer.py...")
    # Run the neural style transfer script
    output_audio_filename = 'output.wav'
    output_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], output_audio_filename)
    subprocess.run(['python', 'NeuralStyleTransfer.py', content_audio_path, style_audio_path])



    # Run the graphs script to generate visualizations and save the image
    output_image_filename = 'output_image.png'
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], output_image_filename)
    print("Running graphs.py...")
    subprocess.run(['python', 'graphs.py', content_audio_path, style_audio_path, output_audio_path, output_image_path], check=True)
    print("graphs.py completed.")
    # Redirect to the result page to display the results
    return redirect(url_for('show_result', audio_filename=output_audio_filename, image_filename=output_image_filename))

@app.route('/result_style')
def show_result():
    audio_filename = request.args.get('audio_filename')
    image_filename = request.args.get('image_filename')
    return render_template('result_style.html', audio_file=audio_filename, image_file=image_filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/classify_audio', methods=['GET', 'POST'])
def classify_audio():
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            return "No file part"
        
        file = request.files['audio_file']
        if file.filename == '':
            return "No selected file"
        
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            try:
                predicted_genre = predict_genre(model, file_path)
                return render_template('classify_audio.html', prediction=predicted_genre)
            except Exception as e:
                print(f"Error: {e}")
                return "Error in processing the file. Prediction failed."
    
    return render_template('classify_audio.html')

@app.route('/audio_visualization')
def audio_visualization():
    return render_template('audio_visualization.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return "No file part"
    file = request.files['audio']
    if file.filename == '':
        return "No selected file"
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Analyze the audio file
    waveform_image_path = generate_waveform(file_path)
    instruments_info = analyze_audio(file_path)

    return render_template('result.html', waveform=waveform_image_path, instruments=instruments_info)

def generate_waveform(file_path):
    y, sr = librosa.load(file_path)
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    waveform_image_path = 'static/waveform.png'
    plt.savefig(waveform_image_path)
    plt.close()
    return waveform_image_path

def analyze_audio(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path)

    # Get pitches and magnitudes
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Get average pitch
    pitch = np.mean(pitches[pitches > 0])

    # Extract spectral features for potential instrument identification
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # Use spectral features to guess instruments (simplified detection logic)
    if spectral_centroid > 3000 and spectral_bandwidth > 2000:
        instruments = 'Electric Guitar'
    elif 1000 < spectral_centroid < 3000:
        instruments = 'Piano'
    elif spectral_centroid < 1000:
        instruments = 'Bass Guitar or Cello'
    else:
        instruments = 'Unknown'

    # Simulate other audio properties
    audio_info = {
        'pitch': pitch,
        'instruments': instruments,
        'duration': librosa.get_duration(y=y, sr=sr),
        'sample_rate': sr
    }

    return audio_info

if __name__ == '__main__':
    app.run(debug=True)
