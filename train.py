import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
import warnings

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


def get_model(config, vocab_src_len, vocab_target_len):
    model = build_transformer(vocab_src_len, vocab_target_len, config["seq_len"], config["seq_len"], config["d_model"])
    return model

def train_model(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        RuntimeError("No GPU found. Please use a GPU to train the Transformer.")
    print(f'Using device: {device}')

    Path(config["model_folder"]).mkdir(exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_source, tokenizer_target = get_dataset(config)
    model = get_model(config, tokenizer_source.get_vocab_size(), tokenizer_target.get_vocab_size()).to(device)

    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps= 1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights(config, config['preload'])
        print(f'Loading weights from {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer'])
        global_step = state['global_step']

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_source.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # (B, S, V)
            encoder_output = model.encoder(encoder_input, encoder_mask)
            decoder_output = model.decoder(encoder_output, encoder_mask, decoder_input, decoder_mask)

            # (B, S, TargetVocab)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device) # (B, S)

            # (B, S, TargetVocab) -> (B*S, TargetVocab)
            loss = criterion(proj_output.view(-1, proj_output.size(-1)), label.view(-1))

            batch_iterator.set_postfix({f'loss': f"{loss.item(): 6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Save model after each epoch. Also save optimizer state, or else the optimizer will have to figure out the state from scratch
        model_filename = get_weights(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)







    
                              
