from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, \
                                    CuDNNGRU, Bidirectional, Dense, \
                                    GlobalAveragePooling1D, GlobalMaxPooling1D,\
                                    Conv1D, LSTM, Add, BatchNormalization,\
                                    Activation, CuDNNLSTM, Dropout, Reshape,\
                                    Conv2D, MaxPooling2D, Flatten, Subtract, \
                                    Softmax, Dot

from custom_layers import Match

CONFIG = {
            'dropout': 0.2,
            'hidden_size': 100,
            'hidden_size_aggregation':100,
            'dense_units': 50
            }

def get_model(max_length=50,
              max_vocab_size=50000,
              embedding_dim=300,
              embedding_weight=None):
    """
        Module with dot attention and GPU Aggregation
    """

    query = Input(shape=(max_length,))
    doc = Input(shape=(max_length,))

    embedding = Embedding(max_vocab_size, embedding_dim,
                          weights=[embedding_weight] if embedding_weight is not None else None,
                          trainable=False if embedding_weight is not None else True)
    q_embed = embedding(query)
    d_embed = embedding(doc)

    q_embed = Dropout(rate=CONFIG['dropout'])(q_embed)
    d_embed = Dropout(rate=CONFIG['dropout'])(d_embed)


    rnn = Bidirectional(CuDNNLSTM(CONFIG['hidden_size'], return_sequences=True))

    q_conv1 = rnn(q_embed)
    d_conv1 = rnn(d_embed)

    cross = Match(match_type='dot')([q_conv1, d_conv1])

    # Attention
    cross = Reshape((max_length, max_length))(cross)
    softmax_cross_doc1 = Softmax(axis=0)(cross)
    attention_outputs = Dot(axes=[1,1])([softmax_cross_doc1,d_conv1])

    aggregation_rnn = Bidirectional(CuDNNLSTM(CONFIG['hidden_size_aggregation'], return_sequences=False))

    # Aggregation for doc1
    aggregation_rnn_input_doc1 = Concatenate(axis=2)([q_conv1, attention_outputs])
    aggregation_rnn_output_doc1 = aggregation_rnn(aggregation_rnn_input_doc1)


    pool1_flat = aggregation_rnn_output_doc1
    pool1_flat = Dense(CONFIG['dense_units'],activation='relu')(pool1_flat)
    pool1_flat_drop = Dropout(rate=CONFIG['dropout'])(pool1_flat)
    out_ = Dense(1,activation='sigmoid')(pool1_flat_drop)

    model = Model(inputs=[query, doc], outputs=out_)
    model.compile(optimizer='adam', loss="binary_crossentropy")
    return model
