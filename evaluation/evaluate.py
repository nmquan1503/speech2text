import torch
from tqdm import tqdm
import jiwer

import config
from data.tokenizer import Tokenizer
from data.dataloader import build_dataloader
from ssm_mamba import ASRModelConfig, ASRModel

def evaluate():
    tokenizer = Tokenizer()
    test_loader = build_dataloader(config.TEST_PREFIX, tokenizer, False)
    model = ASRModel(ASRModelConfig(
        vocab_size=config.VOCAB_SIZE,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        n_features=config.N_MELS,
        model_dim=config.MODEL_DIM,
        state_dim=config.STATE_DIM,
        conv_kernel=config.CONV_KERNEL,
        num_layers=config.NUM_LAYERS,
    ))

    device = next(model.parameters()).device

    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))

    model.eval()

    all_preds = []
    all_refs = []

    for batch in tqdm(test_loader, desc="Test"):
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        feature_lengths = batch["feature_lengths"].to(device)

        seq_ids = model.generate(features, feature_lengths, config.MAX_NEW_TOKENS, config.TEMPERATURE).cpu()
        targets = targets.cpu()

        for pred_ids, tgt_ids in zip(seq_ids, targets):
            pred_ids = pred_ids.tolist()
            tgt_ids = tgt_ids.tolist()

            pred_text = tokenizer.decode(pred_ids)
            tgt_text = tokenizer.decode(tgt_ids)

            all_preds.append(pred_text)
            all_refs.append(tgt_text)

    wer = jiwer.wer(all_refs, all_preds)
    print(f"\nWER: {wer:.4f}")

if __name__ == "__main__":
    evaluate()