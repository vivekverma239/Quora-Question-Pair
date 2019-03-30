# Quora Duplicate Question Pair Challenge

The task is given two pair of sentences check if both mean same thing or not. The dataset
contains 400K question pairs. For validation we randomly sample out 5K examples and
measure performance on that.

## Model
The model is an attention based model. The first step is to pass both the
sentences with a LSTM Layer. The second document's LSTM representation is combined with first document's LSTM representation. For more details, have a look at `model.py`. The attention is similar to Attention Flow Layer (defind [here](https://arxiv.org/pdf/1611.01603.pdf)) with only Context2Query Attention.

## How to run?

```bash
python main.py --data-file DATA_FILE [--epochs EPOCHS]
              [--max-vocab-size MAX_VOCAB_SIZE]
              [--embedding-file-path EMBEDDING_FILE_PATH]
              [--max-length MAX_LENGTH]
              [--embedding-dim EMBEDDING_DIM]
              [--validation-split VALIDATION_SPLIT]
              [--test-split TEST_SPLIT]
              [--data-pickle-file DATA_PICKLE_FILE]
              [--use-pickled-data USE_PICKLED_DATA]
              [--data-aug DATA_AUG]
              [--model-save-filepath MODEL_SAVE_FILEPATH]
```
params
  - data-file: TSV File which contains Quora Data
  - max-vocab-size: Max Vocab size (50000)
  - embedding-file-path: Path to glove embedding ('data/glove/glove.6B.300d.txt')
  - max-length: Max sentence lenth (50)
  - embedding-dim: Embedding Dimension
  - validation-split: Number of Samples for validation (5000)
  - test-split: Number of Samples for validation (5000)
  - data-pickle-file: File where processed data will be cached (data/data.pkl)
  - use-pickled-data: Whether to use cached data if available (True)
  - data-aug: Whether to swich question pairs in dataset
  - model-save-filepath: File where to save keras model

## Results
The model achieves Accuracy of 85.06% on a holdout validation set of 5000 samples. Embedding used for this is glove.840B.300d.
