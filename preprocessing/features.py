import numpy as np
import librosa

import config

def transform(audio: np.ndarray, sampling_rate: int) -> np.ndarray:
    if sampling_rate != config.TARGET_SAMPLING_RATE:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=config.TARGET_SAMPLING_RATE)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=config.TARGET_SAMPLING_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        power=2.0
    )
    logmel = np.log(mel + 1e-6)
    return logmel.T.astype(np.float32)