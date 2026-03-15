import torch

from ssm_mamba import ASRModelConfig, ASRModel
from data.tokenizer import Tokenizer
from data.dataloader import build_dataloader
from training.trainer import Trainer
import config

def train():
    tokenizer = Tokenizer()
    train_loader = build_dataloader(config.TRAIN_PREFIX, tokenizer)
    dev_loader = build_dataloader(config.DEV_PREFIX, tokenizer, False)
    model = ASRModel(ASRModelConfig(
        vocab_size=config.VOCAB_SIZE,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        n_features=config.N_MELS,
        model_dim=config.MODEL_DIM,
        state_dim=config.STATE_DIM,
        conv_kernel=config.CONV_KERNEL,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS
    ))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.bos_id)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        criterion=criterion
    )

    trainer.train()

if __name__ == "__main__":
    train()
