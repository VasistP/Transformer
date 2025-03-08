from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 1e-3,
        "seq_len": 512,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "es",
        "model_folder": "trained_weights",
        "model_basename": "tmodel_",  # Changed from model_filename to model_basename
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "transformer/runs/tmodel",
        "datasource": "opus_books"
    }

def get_weights(config, epoch: str):
    model_folder = config["model_folder"]  # Removed trailing comma
    model_basename = config["model_basename"]  # Removed trailing comma
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path('.') / model_folder / model_filename)