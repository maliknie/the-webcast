import os
import uuid
import re
import tempfile
import platform
import subprocess
import sys
from typing import List, Optional
from dotenv import load_dotenv
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs
from io import BytesIO
import audioop

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Client will be initialized when needed
client = None

# Configuration
DEFAULT_MODELS = {
    "most affordable": "eleven_flash_v2.5",
    "default": "eleven_multilingual_v2",
    "fastest": "eleven_turbo_v2.5",
    "newone": "eleven_v3"
}

DEFAULT_MODEL = DEFAULT_MODELS["default"]

AUDIO_FORMATS = {
    "high_quality": "mp3_44100_192",
    "balanced": "mp3_44100_128",  # default
    "low_latency": "mp3_22050_64",
    "pcm": "pcm_16000"
}

DEFAULT_OUTPUT_FORMAT = AUDIO_FORMATS["balanced"]

# Default chunk size for API limits
DEFAULT_CHUNK_SIZE = 7000

def _play_audio_cross_platform(audio_bytes: bytes, format: str = "mp3") -> bool:
    """
    Cross-platform audio playback that works on macOS, Windows, and Linux.
    Takes raw audio bytes and plays them directly without FFmpeg dependency.
    Returns True if successful, False otherwise.
    """
    try:
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            # Use afplay (built-in macOS audio player)
            with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file.flush()
                result = subprocess.run(["afplay", tmp_file.name], 
                                      capture_output=True, timeout=30)
                os.unlink(tmp_file.name)
                return result.returncode == 0
                
        elif system == "windows":  # Windows
            # Use Windows Media Player (built-in)
            with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file.flush()
                result = subprocess.run(["powershell", "-c", 
                                       f"(New-Object Media.SoundPlayer '{tmp_file.name}').PlaySync();"], 
                                      capture_output=True, timeout=30)
                os.unlink(tmp_file.name)
                return result.returncode == 0
                
        elif system == "linux":  # Linux
            # Try multiple audio players in order of preference
            players = ["paplay", "aplay", "play", "mpv", "vlc"]
            for player in players:
                try:
                    with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp_file:
                        tmp_file.write(audio_bytes)
                        tmp_file.flush()
                        result = subprocess.run([player, tmp_file.name], 
                                              capture_output=True, timeout=30)
                        os.unlink(tmp_file.name)
                        if result.returncode == 0:
                            return True
                except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                    continue
            return False
            
        else:
            print(f"Unsupported platform: {system}")
            return False
            
    except Exception as e:
        print(f"Error playing audio: {e}")
        return False

def _check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available on the system."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False

def _get_client():
    """Initialize and return the ElevenLabs client."""
    global client
    if client is None:
        if not ELEVENLABS_API_KEY:
            raise ValueError("ELEVENLABS_API_KEY not found in environment variables!")
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    return client

def chunk_text_by_words(text: str, chunk_size: int) -> List[str]:
    """Fallback to word-based chunking for very long sentences"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= chunk_size:  # +1 for space
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def chunk_text_by_sentences(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """
    Split text into chunks at sentence boundaries.
    """
    text = text.strip()
    if not text:
        return []

    sentence_endings = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_endings, text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            if len(sentence) > chunk_size:
                chunks.extend(chunk_text_by_words(sentence, chunk_size))
            else:
                current_chunk = sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """Main chunking function that uses sentence boundaries first"""
    return chunk_text_by_sentences(text, chunk_size)

def _convert_chunk_to_mp3_bytes(text_chunk: str, voice_id: str = "pNInz6obpgDQGcFmaJgB", model_id: str = DEFAULT_MODEL, output_format: str = DEFAULT_OUTPUT_FORMAT, retries: int = 1) -> bytes:
    """
    Convert a single chunk to audio bytes with error handling.
    """
    client = _get_client()
    attempt = 0
    while attempt <= retries:
        try:
            resp = client.text_to_speech.convert(
                text=text_chunk,
                voice_id=voice_id,
                model_id=model_id,
                output_format=output_format
            )
            if isinstance(resp, (bytes, bytearray)):
                return bytes(resp)
            else:
                return b"".join(part for part in resp if part)
        except Exception as e:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"Failed to convert chunk to audio after {retries+1} attempts: {e}")
            print(f"Warning: retrying chunk conversion ({attempt}/{retries}) due to error: {e}")

def generate_meaningful_filename(text: str, prefix: str = "webcast") -> str:
    """Generate a meaningful filename based on text content."""
    import re
    from datetime import datetime
    
    # Clean the text for filename
    clean_text = re.sub(r'[^\w\s-]', '', text.lower())
    clean_text = re.sub(r'[-\s]+', '_', clean_text)
    
    # Take first 30 characters and add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_part = clean_text[:30] if clean_text else "response"
    
    return f"{prefix}_{filename_part}_{timestamp}"

def text_to_mp3_file(text: str, out_dir: str, voice_id: str = "pNInz6obpgDQGcFmaJgB", 
                     filename_prefix: Optional[str] = None, model_id: str = DEFAULT_MODEL, 
                     chunk_size: int = DEFAULT_CHUNK_SIZE, bitrate: str = "128k") -> str:
    """
    Convert long text to a single MP3 file with robust error handling.
    """
    # Initialize variables
    out_file = None
    tmp_files = []
    
    try:
        # Set up output path
        os.makedirs(out_dir, exist_ok=True)
        if not filename_prefix:
            filename_prefix = "tts"
        
        # Generate meaningful filename
        meaningful_name = generate_meaningful_filename(text, filename_prefix)
        out_file = os.path.join(out_dir, f"{meaningful_name}.mp3")
        
        print(f"Will save audio to: {out_file}")
        
        # Validate input
        text = text.strip()
        if not text:
            raise ValueError("Empty text provided")
        
        # Process chunks
        chunks = chunk_text(text, chunk_size)
        
        # Generate temporary files for each chunk
        for i, chunk in enumerate(chunks):
            tmp_path = os.path.join(tempfile.gettempdir(), f"eleven_chunk_{uuid.uuid4().hex}_{i}.mp3")
            audio_bytes = _convert_chunk_to_mp3_bytes(chunk, voice_id, model_id=model_id)
            with open(tmp_path, "wb") as tf:
                tf.write(audio_bytes)
            tmp_files.append(tmp_path)
        
        # Combine all chunks
        if not tmp_files:
            raise RuntimeError("No audio chunks were generated")
            
        combined = AudioSegment.from_file(tmp_files[0], format="mp3")
        for path in tmp_files[1:]:
            combined += AudioSegment.from_file(path, format="mp3")
        
        # Export the final file
        combined.export(out_file, format="mp3", bitrate=bitrate)
        
        # CRITICAL: Explicitly return the file path
        print(f"Successfully exported to: {out_file}")
        print(f"About to return: {out_file}")
        return out_file  # <-- This MUST be here!
        
    except Exception as e:
        print(f"Error during MP3 generation: {e}")
        raise  # Re-raise the exception after logging
        
    finally:
        # Clean up temporary files
        # DO NOT put a return statement here unless you really mean it!
        for path in tmp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

def stream_text_to_speech(text: str, voice_id: str = "pNInz6obpgDQGcFmaJgB", model_id: str = DEFAULT_MODEL, chunk_size: int = DEFAULT_CHUNK_SIZE, save_to_file: bool = True, out_file: Optional[str] = None) -> Optional[str]:
    """
    Play long text in chunks sequentially, low latency streaming.
    Optionally save the combined audio to file.
    Uses cross-platform audio playback that works on macOS, Windows, and Linux.
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty text provided")

    chunks = chunk_text(text, chunk_size)
    audio_segments = []

    # Check if FFmpeg is available for better audio processing
    ffmpeg_available = _check_ffmpeg_available()
    if not ffmpeg_available:
        print("Warning: FFmpeg not found. Using cross-platform audio playback.")
        print("For better audio quality, consider installing FFmpeg.")

    for i, chunk in enumerate(chunks, 1):
        print(f"Playing chunk {i}/{len(chunks)}...")
        try:
            audio_bytes = _convert_chunk_to_mp3_bytes(chunk, voice_id, model_id=model_id)
            
            # Use cross-platform audio playback directly with raw bytes
            success = _play_audio_cross_platform(audio_bytes, format="mp3")
            if not success:
                print(f"Warning: Failed to play chunk {i} using cross-platform method")
                # Save the chunk to a temporary file for manual playback
                try:
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                        tmp_file.write(audio_bytes)
                        tmp_file.flush()
                        print(f"Saved chunk {i} to {tmp_file.name} (manual playback required)")
                except Exception as save_error:
                    print(f"Error saving chunk {i}: {save_error}")
            
            # Store audio bytes for potential file saving
            audio_segments.append(audio_bytes)
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")

    if save_to_file and audio_segments:
        if not out_file:
            out_file = os.path.join(tempfile.gettempdir(), f"stream_{uuid.uuid4().hex}.mp3")
        try:
            # Combine all audio bytes into a single file
            with open(out_file, "wb") as f:
                for audio_bytes in audio_segments:
                    f.write(audio_bytes)
            print(f"Combined audio saved to: {out_file}")
            return out_file
        except Exception as e:
            print(f"Error saving combined audio: {e}")
            return None
    return None

def list_saved_files(out_dir: str) -> List[str]:
    """Return list of mp3 files in out_dir sorted by modified time (desc)."""
    if not os.path.isdir(out_dir):
        return []
    files = [f for f in os.listdir(out_dir) if f.lower().endswith(".mp3")]
    files_full = [os.path.join(out_dir, f) for f in files]
    files_full.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files_full

__all__ = [
    'chunk_text_by_sentences',
    'chunk_text_by_words', 
    '_convert_chunk_to_mp3_bytes',
    'stream_text_to_speech',
    'text_to_mp3_file',
    'list_saved_files',
    'DEFAULT_MODEL',
    'DEFAULT_OUTPUT_FORMAT',
    'CHUNK_SIZE'
]
