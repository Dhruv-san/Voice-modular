import os
import glob
import torch
import hashlib
import librosa
import base64
from glob import glob
import numpy as np
from pydub import AudioSegment

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

def split_audio_whisper(audio_path, audio_name, target_dir='processed'):
    if WhisperModel is None:
        raise ImportError("faster-whisper is not installed.")
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    audio = AudioSegment.from_file(audio_path)
    max_len = len(audio)
    target_folder = os.path.join(target_dir, audio_name)
    segments, _ = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    segments = list(segments)    
    os.makedirs(target_folder, exist_ok=True)
    wavs_folder = os.path.join(target_folder, 'wavs')
    os.makedirs(wavs_folder, exist_ok=True)
    for k, w in enumerate(segments):
        start_time = max(0, w.start)
        end_time = w.end
        audio_seg = audio[int(start_time * 1000) : min(max_len, int(end_time * 1000) + 80)]
        fname = f"{audio_name}_seg{k}.wav"
        if audio_seg.duration_seconds > 1.5 and audio_seg.duration_seconds < 20.:
            audio_seg.export(os.path.join(wavs_folder, fname), format='wav')
    return wavs_folder

def split_audio_vad(audio_path, audio_name, target_dir, split_seconds=10.0):
    # Fallback to a simple split if whisper_timestamped is missing or failing
    print("Using simple fallback split instead of VAD")
    audio = AudioSegment.from_file(audio_path)
    audio_dur = audio.duration_seconds
    target_folder = os.path.join(target_dir, audio_name)
    wavs_folder = os.path.join(target_folder, 'wavs')
    os.makedirs(wavs_folder, exist_ok=True)

    num_splits = max(1, int(np.ceil(audio_dur / split_seconds)))
    interval = audio_dur / num_splits

    for i in range(num_splits):
        start_time = i * interval
        end_time = min((i + 1) * interval, audio_dur)
        output_file = f"{wavs_folder}/{audio_name}_seg{i}.wav"
        audio_seg = audio[int(start_time * 1000): int(end_time * 1000)]
        audio_seg.export(output_file, format='wav')
    return wavs_folder

def hash_numpy_array(audio_path):
    array, _ = librosa.load(audio_path, sr=None, mono=True)
    hash_object = hashlib.sha256(array.tobytes())
    return base64.b64encode(hash_object.digest()).decode('utf-8')[:16].replace('/', '_^')

def get_se(audio_path, vc_model, target_dir='processed', vad=True):
    device = vc_model.device
    version = vc_model.version
    audio_name = f"{os.path.basename(audio_path).rsplit('.', 1)[0]}_{version}_{hash_numpy_array(audio_path)}"
    se_path = os.path.join(target_dir, audio_name, 'se.pth')
    
    if vad:
        try:
            # Try to use the original VAD logic if possible, otherwise fallback
            from whisper_timestamped.transcribe import get_audio_tensor, get_vad_segments
            SAMPLE_RATE = 16000
            audio_vad = get_audio_tensor(audio_path)
            segments = get_vad_segments(audio_vad, output_sample=True, method="silero")
            segments = [(float(seg["start"]) / SAMPLE_RATE, float(seg["end"]) / SAMPLE_RATE) for seg in segments]
            audio = AudioSegment.from_file(audio_path)
            audio_active = AudioSegment.silent(duration=0)
            for s, e in segments:
                audio_active += audio[int(s * 1000) : int(e * 1000)]

            wavs_folder = os.path.join(target_dir, audio_name, 'wavs')
            os.makedirs(wavs_folder, exist_ok=True)
            audio_dur = audio_active.duration_seconds
            num_splits = max(1, int(np.round(audio_dur / 10.0)))
            interval = audio_dur / num_splits
            for i in range(num_splits):
                audio_seg = audio_active[int(i*interval*1000): int(min((i+1)*interval, audio_dur)*1000)]
                audio_seg.export(f"{wavs_folder}/{audio_name}_seg{i}.wav", format='wav')
        except Exception:
            wavs_folder = split_audio_vad(audio_path, target_dir=target_dir, audio_name=audio_name)
    else:
        wavs_folder = split_audio_whisper(audio_path, target_dir=target_dir, audio_name=audio_name)
    
    audio_segs = glob(f'{wavs_folder}/*.wav')
    return vc_model.extract_se(audio_segs, se_save_path=se_path), audio_name
