import whisper
import os

# Load the Whisper model
model = whisper.load_model("medium")

def get_audio_content(file_path):
    """
    Transcribe audio content from a given file path using Whisper model.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Transcribe: {file_path}")
    result = model.transcribe(file_path, task="transcribe", fp16=False)
    return result['text']