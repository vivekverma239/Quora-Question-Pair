"""
    WORKS of tensorflow 1.7.0
"""
import tensorflow as tf
from collections import namedtuple
from sklearn.model_selection import train_test_split
import albert.tokenizer as tokenizer
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from albert.helper import convert_examples_to_features
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, \
                                     Bidirectional, Dense, \
                                    GlobalAveragePooling1D, GlobalMaxPooling1D,\
                                    Conv1D, LSTM, Add, BatchNormalization,\
                                    Activation, Dropout, Reshape,\
                                    Conv2D, MaxPooling2D, Flatten, Subtract, \
                                    Softmax, Dot, TimeDistributed, Lambda


from custom_layers import Match

Example = namedtuple('Example', ('question1', 'question2', 'label'), defaults=(None, None, None))

_DEFAULT_BERT_MODEL = 'https://tfhub.dev/google/albert_base/2'
# _DEFAULT_BERT_MODEL = "https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1"
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_MAX_SEQ_LENGTH = 30

class BertModel:
    def __init__(self, bert_model=_DEFAULT_BERT_MODEL,
                 is_bert_trainable=False):
        m = hub.Module(bert_model, trainable=is_bert_trainable)
        self.bert_layer = Lambda(
                    lambda x: m(dict(input_ids=x["input_ids"], input_mask=x["input_mask"], segment_ids=x["segment_ids"]), signature="tokens", as_dict=True)["pooled_output"]
                    )
        # tokenization_info = m.signatures["tokenization_info"]()
        # vocab_file = tokenization_info["vocab_file"].numpy()
        # do_lower_case = tokenization_info["do_lower_case"].numpy()
        # asset_paths = [i.asset_path.numpy().decode('utf-8') for i in m.asset_paths]
        # vocab_file = [i for i in asset_paths if "vocab" in i][0]
        self.tokenizer = tokenizer.FullTokenizer.from_hub_module(_DEFAULT_BERT_MODEL, spm_model_file=None)
        # self.bert_layer = hub.KerasLayer(_DEFAULT_BERT_MODEL, trainable=is_bert_trainable, signature="default")

    def _get_model(self, max_seq_length, num_classes):
        input_question1_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                               name="input_word_ids_q1")
        input_question1_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_mask_q1")
        segment_question1_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                            name="segment_ids_q1")

        input_question2_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                               name="input_word_ids")
        input_question2_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_mask")
        segment_question2_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                            name="segment_ids")

        pooled_q1_output = self.bert_layer(inputs=dict(input_ids=input_question1_ids, input_mask=input_question1_mask, segment_ids=segment_question1_ids))
        # pooled_q1_output, sequence_q1_output = temp["pooled_output"], temp["sequence_output"]
        pooled_q2_output = self.bert_layer(inputs=dict(input_ids=input_question2_ids, input_mask=input_question2_mask, segment_ids=segment_question2_ids))
        # pooled_q2_output, sequence_q2_output = temp["pooled_output"], temp["sequence_output"]

        diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4 * 128 + 2*32,))([pooled_q1_output, pooled_q2_output])
        mul = Lambda(lambda x: x[0] * x[1], output_shape=(4 * 128 + 2*32,))([pooled_q1_output, pooled_q2_output])

        merge = Concatenate()([diff, mul])

        x = Dropout(0.2)(merge)
        x = BatchNormalization()(x)
        x = Dense(300, activation='relu')(x)

        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        pred = Dense(1, activation='sigmoid')(x)

        model = tf.keras.models.Model(inputs=[input_question1_ids, input_question1_mask, 
                            segment_question1_ids, input_question2_ids, 
                            input_question2_mask, segment_question2_ids], outputs=pred)
        return model

    def fit(self, X, y, num_classes, validation_data=None, validation_split=None,
                batch_size=_DEFAULT_BATCH_SIZE,
                max_seq_length=_DEFAULT_MAX_SEQ_LENGTH,
                epochs=1):
        """
            Runs keras training routine after preprocessing the model
            params:
             X - A list of text for which to predict classes
             y - Output classes numerically encoded
        """
        is_test_data_available = False

        # Check if validation split is provided and split accordingly
        if validation_data:
            (train_question1, train_question2), y_train = X, y
            (val_question1, val_question2), y_val = validation_data
            is_test_data_available = True
        elif validation_split:
            train_question1, val_question1, train_question2, val_question2,  y_train, y_val = train_test_split(X[0], X[1], y,
                        test_size=validation_split)
            is_test_data_available = True
        else:
            (train_question1, train_question2), y_train = X, y
            (val_question1, val_question2), y_val = (None, None), None

        train_examples = [Example(*item) for item in zip(train_question1, train_question2, y_train)]
        train_features = convert_examples_to_features(self.tokenizer, train_examples, max_seq_length, is_training=True)

        if is_test_data_available:
            test_examples = [Example(*item) for item in zip(val_question1, val_question2, y_val)]
            test_features = convert_examples_to_features(self.tokenizer, test_examples, max_seq_length, is_training=True)

        model = self._get_model(max_seq_length, num_classes)
        model.compile(loss='sparse_categorical_crossentropy' if num_classes > 1 else "binary_crossentropy", optimizer=tf.keras.optimizers.Adam(2e-5), metrics=['accuracy'])
        if is_test_data_available:
            validation_data = (test_features[:6], test_features[6])
            model.fit(train_features[:6], train_features[6], \
                      validation_data=validation_data, epochs=epochs)
        else:
            model.fit(train_features[:6], train_features[6], num_classes=1, 
                      batch_size=batch_size, epochs=epochs)


def main(data_file, validation_split=5000, test_split=5000):
    data = pd.read_csv(data_file, sep='\t')

    # Shuffle and split dataframe
    np.random.seed(200)
    data.iloc[np.random.permutation(len(data))]
    data = data.iloc[:20000]
    train_df, valid_df, test_df = data.iloc[:-(validation_split+test_split)],\
                                  data.iloc[-(validation_split+test_split):-test_split],\
                                  data.iloc[-test_split:, :]
    
    X = [train_df["question1"].tolist(), train_df["question2"].tolist()]
    y = train_df["is_duplicate"].tolist()
    X_val = [valid_df["question1"].tolist(), valid_df["question2"].tolist()]
    y_val = valid_df["is_duplicate"].tolist()
    model = BertModel()
    tf.keras.backend.get_session().run(tf.global_variables_initializer())
    model.fit(X, y, num_classes=1, validation_data=(X_val, y_val), epochs=10)

if __name__ == '__main__':
    main("data/quora.txt")