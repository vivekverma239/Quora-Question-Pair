import random
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing import text, sequence


def _process_data(dataframe, max_length, vocab):
    """
    
    """


def _load_quora_data(data_file,\
                    max_length=30,
                    max_vocab_size=30000,
                    validation_split=5000,
                    test_split=5000,
                    seed=100,
                    processor_config_filepath='preprocessor.pkl'):
    """
        Load Quora Dataset from TSV file

        :params:
            - data_file: TSV data file provided by Quora
            - max_length: Max Length of Questions, Questions data will be
                                truncated upto this length
            - validation_split: How much to sample for validation
            - test_split: How much to sample for testing
            - seed: Random seed
            - processor_config_filepath: Where to save tokenizer etc 

    """
    # Read data file and assign column names
    data = pd.read_csv(data_file, sep='\t')

    # Shuffle and split dataframe
    np.random.seed(seed)
    data.iloc[np.random.permutation(len(data))]

    train_df, valid_df, test_df = data.iloc[:-(validation_split+test_split)],\
                                  data.iloc[-(validation_split+test_split):-test_split],\
                                  data.iloc[-test_split:, :]

    # Save this for further testing
    data.iloc[-test_split:, :].to_csv("test_data.csv")
    convert_list_to_str = lambda x: list(map(str,x))
    train_question1 = convert_list_to_str(train_df['question1'].tolist())
    train_question2 = convert_list_to_str(train_df['question2'].tolist())
    y_train = train_df['is_duplicate']

    valid_question1 = convert_list_to_str(valid_df['question1'].tolist())
    valid_question2 = convert_list_to_str(valid_df['question2'].tolist())
    y_valid = valid_df['is_duplicate']

    test_question1 = convert_list_to_str(test_df['question1'].tolist())
    test_question2 = convert_list_to_str(test_df['question2'].tolist())
    y_test = test_df['is_duplicate']


    tokenizer = text.Tokenizer(num_words=max_vocab_size)
    tokenizer.fit_on_texts(train_question1 + train_question2 +\
                            valid_question2 + valid_question1 +\
                            test_question2 + test_question1)

    # Processing Training Data
    train_question1 = tokenizer.texts_to_sequences(train_question1)
    train_question2 = tokenizer.texts_to_sequences(train_question2)
    train_question1 = sequence.pad_sequences(train_question1, maxlen=max_length)
    train_question2 = sequence.pad_sequences(train_question2, maxlen=max_length)

    # Processing Validation Datahttps://www.gadgetsnow.com/mobile-phones/Huawei-P30
    valid_question1 = tokenizer.texts_to_sequences(valid_question1)
    valid_question2 = tokenizer.texts_to_sequences(valid_question2)
    valid_question1 = sequence.pad_sequences(valid_question1, maxlen=max_length)
    valid_question2 = sequence.pad_sequences(valid_question2, maxlen=max_length)

    # Processing Test Data
    test_question1 = tokenizer.texts_to_sequences(test_question1)
    test_question2 = tokenizer.texts_to_sequences(test_question2)
    test_question1 = sequence.pad_sequences(test_question1, maxlen=max_length)
    test_question2 = sequence.pad_sequences(test_question2, maxlen=max_length)

    config = {
                'max_length':max_length,
                'tokenizer': tokenizer
             }
    pickle.dump(config, open(processor_config_filepath, "wb"))

    return tokenizer.word_index, train_question1, train_question2, y_train,\
                        valid_question1, valid_question2, y_valid,\
                        test_question1, test_question2, y_test



def load_embedding(embedding_file_path, word_index, embedding_dim):
    """
        Load Embeddings from a text file

        :params:
            - embedding_file_path
            - word_index
            - embedding_dim
    """
    # Create a Numpy Placeholder for Embedding
    max_features = len(word_index)+1
    embedding_weights = np.random.random([max_features, embedding_dim])
    count = 0
    glove_file = open(embedding_file_path)
    for line in glove_file:
        word, vector = line.split(' ')[0], line.split(' ')[1:]
        if word in word_index and word_index[word] <= max_features:
            count += 1
            vector = list(map(float, vector))
            embedding_weights[word_index[word]] = [float(i) for i in vector]

    print('Fraction found in glove {}'.format(count/len(embedding_weights)))
    return embedding_weights
