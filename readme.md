# Transformer-based Neural Machine Translation

This repository contains an implementation of a Transformer model for English to German neural machine translation. It's built using PyTorch and follows the architecture described in the paper "Attention Is All You Need" by Vaswani et al.

## Project Structure

- `config.py` - Configuration settings for the model
- `dataset.py` - Dataset class for loading and preprocessing bilingual data
- `Decoder.py` - Implementation of the Transformer decoder
- `Encoder.py` - Implementation of the Transformer encoder
- `model.py` - Main Transformer model implementation
- `MultiheadAttention.py` - Implementation of multi-head attention mechanism
- `train.py` - Training script for the model

## Features

- Full implementation of the Transformer architecture
- Multi-head attention mechanism
- Positional encoding
- Layer normalization
- Residual connections
- Xavier initialization for parameters

## Requirements

- Python 3.7+
- PyTorch 1.7+
- Hugging Face datasets
- Hugging Face tokenizers
- tqdm
- tensorboard

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/transformer-nmt.git
cd transformer-nmt

# Install dependencies
pip install torch datasets tokenizers tqdm tensorboard
```

## Usage

### Training the Model

To train the model with default settings:

```bash
python train.py
```

The script will:
1. Download the OPUS Books dataset for English-German translation
2. Build tokenizers for both languages
3. Split the data into training and validation sets
4. Train the Transformer model
5. Save model checkpoints after each epoch
6. Log training metrics to tensorboard

### Configuration

You can modify the training parameters in `config.py`:

- `batch_size`: Batch size for training
- `num_epochs`: Number of training epochs
- `lr`: Learning rate
- `seq_len`: Maximum sequence length
- `d_model`: Model dimension
- `lang_src`: Source language code (default: "en")
- `lang_tgt`: Target language code (default: "de")
- `model_folder`: Folder to save model weights
- `preload`: Set to an epoch number to continue training from a checkpoint

### Monitoring Training

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir=transformer/runs
```

## Model Architecture

This implementation includes:

- Encoder: 6 layers of self-attention and feed-forward networks
- Decoder: 6 layers with self-attention, encoder-decoder attention, and feed-forward networks
- Multi-head attention with 8 heads
- Positional encoding using sine and cosine functions
- Residual connections and layer normalization
- Label smoothing during training

## License

[MIT License](LICENSE)

## Acknowledgements

This implementation is inspired by the paper "Attention Is All You Need" by Vaswani et al.
