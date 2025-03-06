import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(
        self, ds, tokenizer_source, tokenizer_target, lang_src, lang_tgt
    ) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt

        self.sos_token = torch.Tensor(
            [tokenizer_source.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.Tensor(
            [tokenizer_source.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.Tensor(
            [tokenizer_source.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair["translation"][self.lang_src]
        tgt_text = src_target_pair["translation"][self.lang_tgt]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding < 0 or dec_num_padding < 0:
            raise ValueError("Sequence length is too long")

        # Adding SOS, EOS and padding tokens
        enc_input = torch.cat(
            [
                self.sos_token,
                torch.Tensor(enc_input_tokens),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding, dtype=torch.int64),
            ]
        )

        # Addding SOS and padding tokens
        dec_input = torch.cat(
            [
                self.sos_token,
                torch.Tensor(dec_input_tokens),
                torch.tensor([self.pad_token] * dec_num_padding, dtype=torch.int64),
            ]
        )

        # Adding EOS and padding tokens to label (expected output from decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding, dtype=torch.int64),
            ]
        )

        assert enc_input.size(0) == dec_input.size(0) == label.size(0) == self.seq_len

        return {
            "encoder_input": enc_input, 
            "decoder_input": dec_input,
            "encoder_mask": (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (dec_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(dec_input.size(0)), #(1, Seq_len) & (1, Seq_len, Seq_len)
            "label": label
            }


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0

