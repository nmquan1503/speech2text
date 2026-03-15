import torch

from ssm_mamba import ASRModelConfig, ASRModel
from data.tokenizer import Tokenizer
from data.dataloader import build_dataloader
from training.trainer import Trainer

def train():

    SPM_PATH = "spm.model"
    TRAIN_PREFIX = "train"
    DEV_PREFIX = "dev"
    BATCH_SIZE = 32

    VOCAB_SIZE = 1000
    MODEL_DIM = 256
    STATE_DIM = 16
    CONV_KERNEL = 4
    NUM_ENCODER_LAYERS = 12
    NUM_DECODER_LAYERS = 2

    LEARNING_RATE = 3e-4

    tokenizer = Tokenizer(
        model_path=SPM_PATH,
        text_path=f"{TRAIN_PREFIX}_texts.txt",
        vocab_size=1000
    )
    train_loader = build_dataloader(TRAIN_PREFIX, tokenizer, BATCH_SIZE)
    dev_loader = build_dataloader(DEV_PREFIX, tokenizer, BATCH_SIZE, False)
    model = ASRModel(ASRModelConfig(
        vocab_size=VOCAB_SIZE,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        n_features=80,
        model_dim=MODEL_DIM,
        state_dim=STATE_DIM,
        conv_kernel=CONV_KERNEL,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS
    ))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.bos_id)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=10,
        save_path="best_model.pt"
    )

    trainer.train()

if __name__ == "__main__":
    train()
