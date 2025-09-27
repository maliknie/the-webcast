import os, io, tempfile, wave, uuid
from typing import Optional, Union
from pathlib import Path
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
client = OpenAI(api_key = OPENAI_API_KEY)

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
MAX_FILE_SIZE_MB = 25
SUPPORTED_FORMATS = ['flac', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'ogg', 'wav', 'webm']

def record_audio_interactive(sample_rate=DEFAULT_SAMPLE_RATE, channels=DEFAULT_CHANNELS):
    # Check if microphone is available first
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if not input_devices:
            raise RuntimeError("No microphone detected")
    except Exception as e:
        print(f"Audio device error: {e}")
        raise
    
    print("Interactive recording mode: Press ENTER to start, ENTER to stop")
    input("Press ENTER to start recording...")
    recorded_chunks = []
    
    def callback(indata, frames, time, status):
        if status:
            print(status)
        recorded_chunks.append(indata.copy())
    
    stream = sd.InputStream(samplerate=sample_rate, channels=channels, dtype=np.int16, callback=callback)
    stream.start()
    try:
        input("Recording... Press ENTER to stop.")
    finally:
        stream.stop()
        stream.close()
    
    audio_data = np.concatenate(recorded_chunks, axis=0)
    temp_file = os.path.join(tempfile.gettempdir(), f"user_recording_{uuid.uuid4().hex}.wav")
    with wave.open(temp_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return temp_file

def record_audio_timed(duration_seconds=5, sample_rate=DEFAULT_SAMPLE_RATE, channels=DEFAULT_CHANNELS):
    print(f"Recording for {duration_seconds} seconds... Speak now!")
    temp_file = os.path.join(tempfile.gettempdir(), f"timed_recording_{uuid.uuid4().hex}.wav")
    audio_data = sd.rec(int(duration_seconds * sample_rate), samplerate=sample_rate, channels=channels, dtype=np.int16)
    sd.wait()
    print("Recording complete!") 
    with wave.open(temp_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return temp_file

def record_audio(duration=None):
    return record_audio_timed(duration) if duration else record_audio_interactive()

def convert_audio_format(input_file: str, output_format: str = "mp3") -> str:
    audio = AudioSegment.from_file(input_file)
    output_file = str(Path(input_file).with_suffix(f".{output_format}"))
    audio.export(output_file, format=output_format)
    return output_file

def transcribe_audio(audio_file_path: str, language: Optional[str] = None) -> str:
    with open(audio_file_path, 'rb') as f:
        params = {"model": "whisper-1", "file": f, "temperature": 0.1}
        if language:
            params["language"] = language
        response = client.audio.transcriptions.create(**params)
    return response.text if hasattr(response, 'text') else str(response)

def transcribe_with_chunks(audio_file_path: str, chunk_duration_ms=60000, overlap_ms=1000) -> str:
    audio = AudioSegment.from_file(audio_file_path)
    total_ms = len(audio)
    chunks = []
    start_ms = 0
    temp_files = []
    
    while start_ms < total_ms:
        end_ms = min(start_ms + chunk_duration_ms, total_ms)
        chunk = audio[start_ms:end_ms]
        temp_file = os.path.join(tempfile.gettempdir(), f"chunk_{uuid.uuid4().hex}.mp3")
        chunk.export(temp_file, format="mp3")
        temp_files.append(temp_file)
        chunks.append(transcribe_audio(temp_file))
        start_ms = end_ms - overlap_ms if end_ms < total_ms else total_ms
    
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
    
    return " ".join(chunks)

def speech_to_text_prompt(input_source: Union[str, int], language: Optional[str] = None) -> str:
    """
    Convert speech to text from either a file or microphone recording.
    
    Args:
        input_source: REQUIRED - Either:
            - str: Path to an existing audio file
            - int: Number of seconds to record from microphone
        language: Optional language code for transcription
    
    Returns:
        Transcribed text
    
    Raises:
        ValueError: If input_source is None or invalid type
    """
    if input_source is None:
        raise ValueError("You must specify either a file path (string) or recording duration (integer)")
    
    if isinstance(input_source, str):
        audio_file = input_source
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"File not found: {audio_file}")
    elif isinstance(input_source, int):
        if input_source <= 0:
            raise ValueError("Recording duration must be positive")
        audio_file = record_audio(input_source)
    else:
        raise ValueError(f"input_source must be string (file path) or integer (recording duration), got {type(input_source)}")
    
    ext = Path(audio_file).suffix.lower().strip('.')
    if ext not in SUPPORTED_FORMATS:
        audio_file = convert_audio_format(audio_file, "mp3")
    
    file_size_mb = os.path.getsize(audio_file) / (1024*1024)
    if file_size_mb > MAX_FILE_SIZE_MB * 0.8:
        return transcribe_with_chunks(audio_file)
    return transcribe_audio(audio_file, language=language)

def create_apertus_prompt_from_speech(audio_source: Union[str, int]) -> str:
    """
    Create AI prompt from speech input.
    
    Args:
        audio_source: REQUIRED - Either:
            - str: Path to audio file
            - int: Seconds to record from microphone
    """
    if audio_source is None:
        raise ValueError("You must specify either a file path or recording duration")
    
    try:
        return speech_to_text_prompt(audio_source)
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        raise

if __name__ == "__main__":
    text_prompt = create_apertus_prompt_from_speech(5)
    print("\nAI Prompt Text:\n", text_prompt)
