from torch.utils.data import Dataset
import numpy as np
import torch

class ASRDataset(Dataset):
    def __init__(self, prefix, tokenizer):
        self.features = np.memmap(f"{prefix}_features", dtype=np.float32, mode="r")
        self.offsets = np.load(f"{prefix}_offsets.npy")
        self.lengths = np.load(f"{prefix}_lengths.npy")
        self.num_features = int((self.offsets[1] - self.offsets[0]) // self.lengths[0])

        with open(f"{prefix}_texts.txt") as f:
            texts = [line.strip() for line in f]
        
        self.targets = [tokenizer.encode(text) for text in texts]
    
    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        offset = self.offsets[index]
        length = self.lengths[index]

        feat = self.features[offset : offset + length * self.num_features]
        feat = feat.reshape(length, self.num_features)
        tgt = self.targets[index]

        return {
            "feature": torch.from_numpy(feat),
            "target": torch.tensor(tgt, dtype=torch.long)
        }

