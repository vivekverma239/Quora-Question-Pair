"""
    Main module for training Quora Duplication Question Detection Model
"""
import os
import pickle
import fire
import numpy as np

from data_loader import _load_quora_data, load_embedding
from evaluation import f1
from models import get_model
from keras.callbacks import EarlyStopping, ModelCheckpoint





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
         processor_config_filepath='data/processor.pkl'
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
        test_question1, test_question2, y_test = pickle.load(open(data_pickle_file, 'rb'))

    else:
        # Load and process all the data
        word_index, train_question1, train_question2, y_train,\
        valid_question1, valid_question2, y_valid,\
        test_question1, test_question2, y_test = _load_quora_data(
                                                    data_file=data_file,
                                                    max_length=max_length,
                                                    max_vocab_size=max_vocab_size,
                                                    validation_split=validation_split,
                                                    test_split=test_split,
                                                    processor_config_filepath=processor_config_filepath
                                                    )

        pickle.dump(
                     [word_index, train_question1, train_question2,
                     y_train, valid_question1, valid_question2,
                     y_valid, test_question1, test_question2,\
                     y_test],
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
    model = get_model(max_length=max_length,
                      max_vocab_size=max_vocab_size,
                      embedding_dim=300,
                      embedding_weight=embedding_weight)

    if data_aug:
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


if __name__ == '__main__':
    fire.Fire(main)
