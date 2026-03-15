import torch
from tqdm import tqdm

import config

class Trainer:
    def __init__(
        self, 
        model,
        train_loader,
        dev_loader,
        optimizer, 
        criterion
    ):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.optimizer = optimizer
        self.criterion = criterion

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)

        self.best_dev_loss = float("inf")
    
    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Train"):
            self.optimizer.zero_grad()
            
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            feature_lengths = batch["feature_lengths"].to(self.device)

            decoder_input = targets[:, :-1]
            labels = targets[:, 1:]

            logits = self.model(features, feature_lengths, decoder_input)

            loss = self.criterion(
                logits.view(-1, config.VOCAB_SIZE),
                labels.view(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _eval(self):
        self.model.eval()
        total_loss = 0.0
        for batch in tqdm(self.dev_loader, desc="Eval"):
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            feature_lengths = batch["feature_lengths"].to(self.device)

            decoder_input = targets[:, :-1]
            labels = targets[:, 1:]

            logits = self.model(features, feature_lengths, decoder_input)
            
            loss = self.criterion(
                logits.reshape(-1, config.VOCAB_SIZE),
                labels.reshape(-1)
            )

            total_loss += loss.item()
        
        return total_loss / len(self.dev_loader)

    def train(self):
        for epoch in range(1, config.NUM_EPOCHS + 1):
            print("=" * 10 + f" Epoch {epoch} " + "=" * 10)
            
            train_loss = self._train_one_epoch()
            dev_loss = self._eval()
            print(f"Train loss: {train_loss:.4f}")
            print(f"Dev loss: {dev_loss:.4f}")

            if dev_loss < self.best_dev_loss:
                self.best_dev_loss = dev_loss
                torch.save(self.model.state_dict(), config.MODEL_PATH)
                print(">>> Save best model")