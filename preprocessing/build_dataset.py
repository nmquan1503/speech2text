import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from preprocessing.features import transform
import config

def build(splits, prefix):
    offsets = []
    lengths = []
    offset = 0

    with open(f"{prefix}_features", "wb") as ff, open(f"{prefix}_texts.txt", "w") as ft:
        for split in splits:
            ds = load_dataset(
                "openslr/librispeech_asr",
                "all",
                split=split,
                streaming=True
            )

            for sample in tqdm(ds):
                audio = sample["audio"]["array"]
                sampling_rate = sample["audio"]["sampling_rate"]

                feat = transform(audio, sampling_rate)  # (lengths, num_features)

                offsets.append(offset)
                lengths.append(feat.shape[0])

                ff.write(feat.tobytes())
                ft.write(sample["text"] + "\n")
    
                offset += feat.size

    np.save(f"{prefix}_offsets.npy", np.array(offsets, dtype=np.int64))
    np.save(f"{prefix}_lengths.npy", np.array(lengths, dtype=np.int32))

if __name__ == "__main__":
    build(config.TRAIN_SPLITS, config.TRAIN_PREFIX)
    build(config.DEV_SPLITS, config.DEV_PREFIX)