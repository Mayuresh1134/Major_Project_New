import os
import uuid
from midi2audio import FluidSynth

# Configure upload and output folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/audio'
ALLOWED_EXTENSIONS = {'mid', 'midi'}

# SoundFont paths
SOUNDFONTS = {
    "piano": "static/soundfonts/Dore Mark's NY S&S Model B-v5.2.sf2",
    "guitar": "static/soundfonts/ChateauGrand-Plus-Instruments-bs16i-v4.sf2",  # Replace with the actual file path
    "drums": "static/soundfonts/Nice-Bass-Plus-Drums-v5.3.sf2"  # Replace with the actual file path
}



# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_midi_to_wav(file, instrument):
    """
    Convert an uploaded MIDI file to WAV using the specified instrument.
    Args:
        file: The uploaded file object.
        instrument: The selected instrument for conversion.

    Returns:
        Tuple (output_filename, error_message).
    """
    # Save the uploaded MIDI file
    midi_filename = f"{uuid.uuid4().hex}.mid"
    midi_path = os.path.join(UPLOAD_FOLDER, midi_filename)
    file.save(midi_path)

    # Generate output WAV filename
    output_filename = f"{uuid.uuid4().hex}.wav"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    # Select the appropriate SoundFont
    soundfont_path = SOUNDFONTS.get(instrument)
    if not soundfont_path or not os.path.exists(soundfont_path):
        return None, f"SoundFont for {instrument} not found!"

    # Convert MIDI to WAV
    try:
        fs = FluidSynth(soundfont_path)
        fs.midi_to_audio(midi_path, output_path)
    except Exception as e:
        return None, f"Error converting MIDI to WAV: {e}"

    return output_filename, None
