import numpy as np
import librosa

TARGET_SAMPLING_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80

def transform(audio: np.ndarray, sampling_rate: int) -> np.ndarray:
    if sampling_rate != TARGET_SAMPLING_RATE:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=TARGET_SAMPLING_RATE)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SAMPLING_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )
    logmel = np.log(mel + 1e-6)
    return logmel.T.astype(np.float32)