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
        # print(item)
        yield item['translation'][lang]


def build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_sentences(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# def clean_text(text):
#     text = text.split("Source:")[0].strip()  # Remove metadata if present
#     return text if text else "[UNK]"

# def get_dataset(config):
#     data_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

#     print("Sample dataset item before preprocessing:", data_raw[0])

#     # Remove dataset entries missing 'translation' key or expected language data
#     cleaned_data = [
#         item for item in data_raw
#         if "translation" in item and config["lang_src"] in item["translation"] and config["lang_tgt"] in item["translation"]
#     ]

#     if not cleaned_data:
#         raise ValueError("Dataset is empty after filtering. Ensure dataset format is correct.")

#     print(f"Dataset size after filtering: {len(cleaned_data)}")

#     # Build tokenizers
#     tokenizer_source = build_tokenizer(config, cleaned_data, config["lang_src"])
#     tokenizer_target = build_tokenizer(config, cleaned_data, config["lang_tgt"])

#     # Perform train-validation split
#     train_size = int(0.8 * len(cleaned_data))
#     val_size = len(cleaned_data) - train_size
#     train_raw, val_raw = random_split(cleaned_data, [train_size, val_size])

#     # Convert Subset objects back to lists
#     train_raw = [cleaned_data[i] for i in train_raw.indices]
#     val_raw = [cleaned_data[i] for i in val_raw.indices]

#     train_ds = BilingualDataset(train_raw, tokenizer_source, tokenizer_target, config["lang_src"], config["lang_tgt"], config["seq_len"])
#     val_ds = BilingualDataset(val_raw, tokenizer_source, tokenizer_target, config["lang_src"], config["lang_tgt"], config["seq_len"])

#     train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
#     val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

#     return train_dataloader, val_dataloader, tokenizer_source, tokenizer_target


def get_dataset(config):
    # data_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}')
    data_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    print("Sample dataset item:", data_raw[0])
    ##############################################
    # data_raw = data_raw['train']
    ##############################################

    # for item in data_raw:
    #     if "translation" in item:
    #         item["translation"][config["lang_src"]] = clean_text(item["translation"].get(config["lang_src"], ""))
    #         item["translation"][config["lang_tgt"]] = clean_text(item["translation"].get(config["lang_tgt"], ""))

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
        # print(f"DEBUG item: {item}")
        src_ids = tokenizer_source.encode(item['translation'][config["lang_src"]]).ids
        tgt_ids = tokenizer_target.encode(item['translation'][config["lang_tgt"]]).ids
        # src_ids = item['encoder_input'].tolist()
        # tgt_ids = item['decoder_input'].tolist()
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max source length: {max_len_src}")
    print(f"Max target length: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_source, tokenizer_target


def get_model(config, vocab_src_len, vocab_target_len):
    model = build_transformer(vocab_src_len, vocab_target_len, config["seq_len"], config["seq_len"], d_model=config["d_model"])
    return model

def train_model(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError("No GPU found. Please use a GPU to train the Transformer.")
    print(f'Using device: {device}')

    # Path(config["model_folder"]).mkdir(exist_ok=True)
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

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

    # preload = config['preload']
    # model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    # if model_filename:
    #     print(f'Preloading model {model_filename}')
    #     state = torch.load(model_filename)
    #     model.load_state_dict(state['model_state_dict'])
    #     initial_epoch = state['epoch'] + 1
    #     optimizer.load_state_dict(state['optimizer_state_dict'])
    #     global_step = state['global_step']
    # else:
    #     print('No model to preload, starting from scratch')

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_source.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)


            # encoder_input = model.src_embed(encoder_input)  # Convert token IDs to embeddings
            # decoder_input = model.target_embedding(decoder_input)

            # (B, S, V)
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

            # (B, S, TargetVocab)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device) # (B, S)

            # (B, S, TargetVocab) -> (B*S, TargetVocab)
            loss = criterion(proj_output.view(-1, tokenizer_target.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({f'loss': f"{loss.item(): 6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

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







    
                              
