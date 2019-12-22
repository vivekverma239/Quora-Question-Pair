import numpy as np
from tqdm import tqdm

def convert_single_example(tokenizer, text, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    tokens_a = tokenizer.tokenize(text)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def convert_examples_to_features( tokenizer, examples, max_seq_length,
                                    is_training):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_question1_ids, input_question1_masks, segment_question1_ids = [], [], []
    input_question2_ids, input_question2_masks, segment_question2_ids = [], [], []
    labels = []
    for example in tqdm(examples, desc="Converting examples to features"):
        input_question1_id, input_question1_mask, segment_question1_id = convert_single_example(
            tokenizer, str(example.question1), max_seq_length
        )
        input_question2_id, input_question2_mask, segment_question2_id = convert_single_example(
            tokenizer, str(example.question2), max_seq_length
        )
        input_question1_ids.append(input_question1_id)
        input_question1_masks.append(input_question1_mask)
        segment_question1_ids.append(segment_question1_id)
        input_question2_ids.append(input_question2_id)
        input_question2_masks.append(input_question2_mask)
        segment_question2_ids.append(segment_question2_id)
        labels.append(example.label)
    return (
        np.array(input_question1_ids),
        np.array(input_question1_masks),
        np.array(segment_question1_ids),
        np.array(input_question2_ids),
        np.array(input_question2_masks),
        np.array(segment_question2_ids),
        np.array(labels).reshape(-1, 1),
    )
