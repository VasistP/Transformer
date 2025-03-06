import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exits(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["UNK", "PAD", "SOS", "EOS"], min_frequency = 2)
        tokenizer.train_from_iterator(get_sentences(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config):
    data_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}')

    # Build tokenizers
    tokenizer_source = build_tokenizer(config, data_raw, config["lang_src"])
    tokenizer_target = build_tokenizer(config, data_raw, config["lang_tgt"])

    train_size = int(0.8 *len(data_raw))
    val_size = len(data_raw) - train_size

    train_raw, val_raw = random_split(data_raw, [train_size, val_size])

    train_ds = BilingualDataset(train_raw, tokenizer_source, tokenizer_target, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_raw, tokenizer_source, tokenizer_target, config["lang_src"], config["lang_tgt"], config["seq_len"])

    max_len_src = 0
    max_len_tgt = 0

    for item in train_ds:
        src_ids = tokenizer_source.encode(item['translation'][config["lang_src"]]).ids
        tgt_ids = tokenizer_target.encode(item['translation'][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max source length: {max_len_src}")
    print(f"Max target length: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_source, tokenizer_target



