import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(
        self, ds, tokenizer_source, tokenizer_target, lang_src, lang_tgt, seq_len):
        super().__init__()

        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt

        self.sos_token = torch.tensor([tokenizer_target.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_target.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_target.token_to_id("[PAD]")], dtype=torch.int64)
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        
        # Check if 'translation' exists in item
        if 'translation' not in item:
            raise KeyError(f"Item at index {idx} does not have 'translation' key")
            
        # Check if source and target languages exist in translation
        if self.lang_src not in item['translation'] or self.lang_tgt not in item['translation']:
            raise KeyError(f"Missing language keys in item at index {idx}")
            
        src_text = item['translation'][self.lang_src]
        tgt_text = item['translation'][self.lang_tgt]

        # Clean source text if needed (remove metadata)
        if "Source:" in src_text:
            src_text = src_text.split("Source:")[0].strip()
            if not src_text:
                src_text = "[UNK]"
                
        # Clean target text if needed
        if "Source:" in tgt_text:
            tgt_text = tgt_text.split("Source:")[0].strip()
            if not tgt_text:
                tgt_text = "[UNK]"

        # Tokenize the texts
        enc_input_tokens = self.tokenizer_source.encode(src_text).ids
        dec_input_tokens = self.tokenizer_target.encode(tgt_text).ids

        # Truncate if too long
        if len(enc_input_tokens) > self.seq_len - 2:  # -2 for SOS and EOS
            enc_input_tokens = enc_input_tokens[:self.seq_len - 2]
            
        if len(dec_input_tokens) > self.seq_len - 2:  # -2 for SOS and EOS
            dec_input_tokens = dec_input_tokens[:self.seq_len - 2]

        # Calculate padding
        enc_num_padding = self.seq_len - len(enc_input_tokens) - 2  # -2 for SOS and EOS tokens
        dec_num_padding = self.seq_len - len(dec_input_tokens) - 1  # -1 for SOS token

        # Adding SOS, EOS and padding tokens
        enc_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding, dtype=torch.int64),
            ],
            dim=0
        )

        # Adding SOS and padding tokens
        dec_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding, dtype=torch.int64),
            ],
            dim=0
        )

        # Adding EOS and padding tokens to label (expected output from decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding, dtype=torch.int64),
            ],
            dim=0
        )

        assert enc_input.size(0) == self.seq_len
        assert dec_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": enc_input, 
            "decoder_input": dec_input,
            "encoder_mask": (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (dec_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(dec_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
            "translation": item['translation']  # Keep original for debugging
        }


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0