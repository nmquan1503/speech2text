import sentencepiece as spm
from pathlib import Path

def train_tokenizer(
    text_path: str,
    model_prefix: str, 
    vocab_size: int = 1000
):
    spm.SentencePieceTrainer.Train(
        input=text_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=-1,
        minloglevel=2
    )

class Tokenizer:
    def __init__(
        self, 
        model_path: str, 
        text_path: str | None = None, 
        vocab_size: int = 1000
    ):
        if not Path(model_path).exists():
            if text_path is None:
                raise ValueError("text_path required to train tokenizer")
            model_prefix = model_path.split(".model")[0]
            train_tokenizer(text_path, model_prefix, vocab_size)
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

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
