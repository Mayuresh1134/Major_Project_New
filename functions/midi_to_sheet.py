import os
import subprocess
import fitz  # PyMuPDF

# Path to MuseScore executable
MUSESCORE_PATH = "C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe"  # Update this as needed

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'mid', 'midi'}

# Function to configure upload folder, output folder and create directories if they don't exist
def setup_directories(app):
    UPLOAD_FOLDER = 'uploads'
    OUTPUT_FOLDER = 'outputs'
    
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

    # Create directories if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to convert MIDI to PDF using MuseScore
def convert_midi_to_pdf(midi_path, output_pdf_path):
    command = [MUSESCORE_PATH, midi_path, '-o', output_pdf_path]
    subprocess.run(command, check=True)

# Function to convert PDF to PNG
def convert_pdf_to_png(output_pdf_path, output_png_path):
    doc = fitz.open(output_pdf_path)
    for page in doc:  # Process all pages if needed
        pix = page.get_pixmap()
        pix.save(output_png_path)  # Save PNG
        break  # Remove this if you want all pages as images
