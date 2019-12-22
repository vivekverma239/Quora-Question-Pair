"""
    Main module for training Quora Duplication Question Detection Model
"""
import os
import pickle
import fire
import numpy as np
import pandas as pd 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from evaluation import f1
from models import  get_model_v1, get_model_v2, get_model_v3
from data_utils import TextProcessor

def load_embedding(embedding_file_path, word_index, embedding_dim):
    """
        Load Embeddings from a text file

        :params:
            - embedding_file_path: Glove embedding filepath (Any embedding in glove format)
            - word_index: Word to index mapping 
            - embedding_dim: Dimentions 
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


def _load_and_process_data(data_file, 
                            max_length=30,
                            max_vocab_size=100000,
                            validation_split=5000,
                            test_split=5000,
                            seed=200,
                            tokenizer_path="data/tokenizer.json"):
                
    data = pd.read_csv(data_file, sep='\t')

    # Shuffle and split dataframe
    np.random.seed(seed)
    data.iloc[np.random.permutation(len(data))]

    train_df, valid_df, test_df = data.iloc[:-(validation_split+test_split)],\
                                  data.iloc[-(validation_split+test_split):-test_split],\
                                  data.iloc[-test_split:, :]

    processor = TextProcessor(num_words=max_vocab_size, tokenizer="nltk", max_length=max_length)
    
    convert_list_to_str = lambda x: list(map(str,x))
    processor.fit(convert_list_to_str(train_df["question1"].tolist() + train_df["question2"].tolist()))
    processor.dump(tokenizer_path)
    train_question1 = processor.process(convert_list_to_str(train_df['question1'].tolist()))
    train_question2 = processor.process(convert_list_to_str(train_df['question2'].tolist()))
    y_train = train_df['is_duplicate']
    valid_question1 = processor.process(convert_list_to_str(valid_df['question1'].tolist()))
    valid_question2 = processor.process(convert_list_to_str(valid_df['question2'].tolist()))
    y_valid = valid_df['is_duplicate']
    test_question1 = processor.process(convert_list_to_str(test_df['question1'].tolist()))
    test_question2 = processor.process(convert_list_to_str(test_df['question2'].tolist()))
    y_test = test_df['is_duplicate']
    
    return  processor.tokenizer.word_index, train_question1, train_question2, y_train,\
                        valid_question1, valid_question2, y_valid,\
                        test_question1, test_question2, y_test, test_df

def main(data_file,
         epochs=20,
         max_vocab_size=50000,
         embedding_file_path='data/glove/glove.6B.300d.txt',
         max_length=50,
         embedding_dim=300,
         validation_split=5000, # Number of Pairs in Validation set
         test_split=5000, # Number of Pairs in test set
         data_pickle_file='data/data.pkl',
         use_pickled_data=True,
         data_aug=False,
         model_save_filepath='data/model.h5',
         tokenizer_path='data/tokenizer.json'
         ):
    """
        Function for model training

        :params:
            - train_tsv_file
            - test_tsv_file
            - max_vocab_size
            - embedding_file_path
            - max_query_length
            - max_response_length
            - embedding_dim
            - validation_split
    """

    word_index = None
    if use_pickled_data and os.path.exists(data_pickle_file):
        print("Loading Pickled Data...")
        word_index, train_question1, train_question2, y_train,\
        valid_question1, valid_question2, y_valid,\
        test_question1, test_question2, y_test, test_df = pickle.load(open(data_pickle_file, 'rb'))
    else:
        # Load and process all the data
        word_index, train_question1, train_question2, y_train,\
        valid_question1, valid_question2, y_valid,\
        test_question1, test_question2, y_test, test_df = _load_and_process_data(
                                                        data_file=data_file,
                                                        max_length=max_length,
                                                        max_vocab_size=max_vocab_size,
                                                        validation_split=validation_split,
                                                        test_split=test_split,
                                                        tokenizer_path=tokenizer_path
                                                    )
        pickle.dump(
                     [word_index, train_question1, train_question2,
                     y_train, valid_question1, valid_question2,
                     y_valid, test_question1, test_question2,\
                     y_test, test_df],
                     open(data_pickle_file, "wb")
                   )

    # Limit word Vocab to Max Vocab
    word_index = {k:v for k, v in word_index.items() if v < max_vocab_size}

    # Case to handle when len(word_index) < max_vocab_size
    max_vocab_size = len(word_index)+1

    # Embedding Loader
    embedding_weight = load_embedding(embedding_file_path,
                                      word_index,
                                      embedding_dim)

    # Define the model
    model = get_model_v3(max_length=max_length,
                      max_vocab_size=max_vocab_size,
                      embedding_dim=embedding_dim,
                      embedding_weight=embedding_weight)

    if data_aug:
        # Augumentation: reverse Q1 and Q2 ordering 
        train_question_aug1 = np.concatenate([train_question1, train_question2], axis=0)
        train_question_aug2 = np.concatenate([train_question2, train_question1], axis=0)
        y_train = np.concatenate([y_train, y_train], axis=0)
        train_question1, train_question2 = train_question_aug1, train_question_aug2

    callbacks =  [EarlyStopping(monitor='val_acc', patience=2),
                  ModelCheckpoint(model_save_filepath, monitor='val_acc', save_best_only=True)]

    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[f1, 'acc'])

    model.fit([train_question1, train_question2], y_train,\
              epochs=epochs,
              batch_size=64,
              validation_data=([valid_question1, valid_question2], y_valid),
              callbacks=callbacks)
              
    test_pred = model.predict([test_question1, test_question2])
    test_df["prediction"] = test_pred
    test_df.to_csv("predictions.csv")


if __name__ == '__main__':
    fire.Fire(main)
