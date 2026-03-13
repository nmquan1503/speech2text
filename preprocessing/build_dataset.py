import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import argparse
import os

from preprocessing.features import transform

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_splits",
        nargs="+",
        default=["train.clean.100"]
    )

    parser.add_argument(
        "--train_dir",
        type=str,
        default=""
    )

    parser.add_argument(
        "--dev_splits",
        nargs="+",
        default=["validation.clean"]
    )

    parser.add_argument(
        "--dev_dir",
        type=str,
        default=""
    )

    return parser.parse_args()

def build(splits, out_prefix):
    offsets = []
    lengths = []
    offset = 0

    with open(f"{out_prefix}_features", "wb") as ff, open(f"{out_prefix}_texts.txt", "w") as ft:
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
                byte_data = feat.tobytes()
                data_size = len(byte_data)

                offsets.append(offset)
                lengths.append(feat.shape[0])

                ff.write(byte_data)
                ft.write(sample["text"] + "\n")
    
                offset += data_size

    np.save(f"{out_prefix}_offsets.npy", np.array(offsets, dtype=np.int64))
    np.save(f"{out_prefix}_lengths.npy", np.array(lengths, dtype=np.int32))

if __name__ == "__main__":
    args = parse_args()

    build(args.train_splits, os.path.join(args.train_dir, "train"))
    build(args.dev_splits, os.path.join(args.dev_dir, "dev"))