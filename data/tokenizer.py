import sentencepiece as spm
from pathlib import Path

import config

def train_tokenizer():
    spm.SentencePieceTrainer.Train(
        input=f"{config.TRAIN_PREFIX}_texts.txt",
        model_prefix=config.SPM_MODEL_PATH.split(".model")[0],
        vocab_size=config.VOCAB_SIZE,
        model_type="unigram",
        character_coverage=1.0,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=-1,
        minloglevel=2
    )

class Tokenizer:
    def __init__(self):
        if not Path(config.SPM_MODEL_PATH).exists():
            train_tokenizer()
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(config.SPM_MODEL_PATH)

        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
    
    def encode(self, text, add_bos=True, add_eos=True):
        ids = self.sp.Encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        
        return ids

    def decode(self, ids):
        return self.sp.Decode(ids)
