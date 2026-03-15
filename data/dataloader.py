from torch.utils.data import DataLoader
import torch

from data.dataset import ASRDataset

def collate_fn(batch):
    features = [item["feature"] for item in batch]
    targets = [item["target"] for item in batch]

    feature_lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)

    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

    target_mask = targets != 0
    
    return {
        "features": features,
        "targets": targets,
        "feature_lengths": feature_lengths,
        "target_mask": target_mask
    }

def build_dataloader(
    prefix,
    tokenizer,
    batch_size=32,
    shuffle=True
):
    dataset = ASRDataset(prefix, tokenizer)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=collate_fn
    )