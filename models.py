from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, \
                                     Bidirectional, Dense, \
                                    GlobalAveragePooling1D, GlobalMaxPooling1D,\
                                    Conv1D, LSTM, Add, BatchNormalization,\
                                    Activation, Dropout, Reshape,\
                                    Conv2D, MaxPooling2D, Flatten, Subtract, \
                                    Softmax, Dot, TimeDistributed, Lambda

from tensorflow.compat.v1.keras.layers import CuDNNLSTM, CuDNNGRU

from custom_layers import Match

CONFIG = {
            'dropout': 0.2,
            'hidden_size': 100,
            'hidden_size_aggregation':100,
            'dense_units': 50
            }

def get_model_v1(max_length,
          max_vocab_size,
          embedding_dim=300,
          embedding_weight=None,
          ):

    query = Input(shape=(max_length,))
    doc = Input(shape=(max_length,))

    embedding = Embedding(max_vocab_size, 300,
                          weights=[embedding_weight] if embedding_weight is not None else None,
                          trainable=True)
    q_embed = embedding(query)
    d_embed = embedding(doc)
    # q_embed = Dropout(rate=0.5)(q_embed)
    # d_embed = Dropout(rate=0.5)(d_embed)
    rnn = Bidirectional(CuDNNLSTM(50, return_sequences=True))

    q_conv1 = rnn(q_embed)
    d_conv1 = rnn(d_embed)
    q_conv1 = Dropout(0.5)(q_conv1)
    d_conv1 = Dropout(0.5)(d_conv1)

    cross = Match(match_type='dot')([q_conv1, d_conv1])

    # z = Reshape((15, 50, 1))(cross)
    # z = Conv2D(filters=50, kernel_size=(3, 3), padding='same', activation='relu')(z)
    # z = Conv2D(filters=25, kernel_size=(3, 3), padding='same', activation='relu')(z)
    # z = MaxPooling2D(pool_size=(3, 3))(z)
    # z = Conv2D(filters=10, kernel_size=(3, 3), padding='same', activation='relu')(z)
    # z = MaxPooling2D(pool_size=(3, 3))(z)

    pool1_flat = Flatten()(cross)
    # pool1_flat = Concatenate()([q_conv1, d_conv1])
    pool1_flat_drop = Dropout(rate=0.5)(pool1_flat)
    out_ = Dense(1, activation="sigmoid")(pool1_flat_drop)

    model = Model(inputs=[query,doc], outputs=out_)
    # model.compile(optimizer='adadelta', loss=rank_hinge_loss())
    return model

def get_model_v2(max_length=50,
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

def get_model_v3(max_length=60,
              max_vocab_size=60000,
              embedding_dim=300,
              embedding_weight=None):
    
    # The embedding layer containing the word vectors
    emb_layer = Embedding(
        input_dim=max_vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_weight],
        input_length=max_length,
        trainable=True
    )
    
    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')

    # Define inputs
    seq1 = Input(shape=(max_length,))
    seq2 = Input(shape=(max_length,))

    # Run inputs through embedding
    emb1 = emb_layer(seq1)
    emb2 = emb_layer(seq2)

    # lstm =  Bidirectional(CuDNNLSTM(100, return_sequences=True))
    # emb1 = lstm(emb1)
    # emb2 = lstm(emb2)

    # Run through CONV + GAP layers
    conv1a = conv1(emb1)
    glob1a = GlobalAveragePooling1D()(conv1a)
    conv1b = conv1(emb2)
    glob1b = GlobalAveragePooling1D()(conv1b)

    conv2a = conv2(emb1)
    glob2a = GlobalAveragePooling1D()(conv2a)
    conv2b = conv2(emb2)
    glob2b = GlobalAveragePooling1D()(conv2b)

    conv3a = conv3(emb1)
    glob3a = GlobalAveragePooling1D()(conv3a)
    conv3b = conv3(emb2)
    glob3b = GlobalAveragePooling1D()(conv3b)

    conv4a = conv4(emb1)
    glob4a = GlobalAveragePooling1D()(conv4a)
    conv4b = conv4(emb2)
    glob4b = GlobalAveragePooling1D()(conv4b)

    conv5a = conv5(emb1)
    glob5a = GlobalAveragePooling1D()(conv5a)
    conv5b = conv5(emb2)
    glob5b = GlobalAveragePooling1D()(conv5b)

    conv6a = conv6(emb1)
    glob6a = GlobalAveragePooling1D()(conv6a)
    conv6b = conv6(emb2)
    glob6b = GlobalAveragePooling1D()(conv6b)

    mergea = Concatenate()([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])
    mergeb = Concatenate()([glob1b, glob2b, glob3b, glob4b, glob5b, glob6b])

    # We take the explicit absolute difference between the two sentences
    # Furthermore we take the multiply different entries to get a different measure of equalness
    diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4 * 128 + 2*32,))([mergea, mergeb])
    mul = Lambda(lambda x: x[0] * x[1], output_shape=(4 * 128 + 2*32,))([mergea, mergeb])

    # # Add the magic features
    # magic_input = Input(shape=(5,))
    # magic_dense = BatchNormalization()(magic_input)
    # magic_dense = Dense(64, activation='relu')(magic_dense)

    # # Add the distance features (these are now TFIDF (character and word), Fuzzy matching, 
    # # nb char 1 and 2, word mover distance and skew/kurtosis of the sentence vector)
    # distance_input = Input(shape=(20,))
    # distance_dense = BatchNormalization()(distance_input)
    # distance_dense = Dense(128, activation='relu')(distance_dense)

    # # Merge the Magic and distance features with the difference layer
    # merge = concatenate([diff, mul, magic_dense, distance_dense])
    merge = Concatenate()([diff, mul])


    # The MLP that determines the outcome
    x = Dropout(0.2)(merge)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    pred = Dense(1, activation='sigmoid')(x)

    # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
    model = Model(inputs=[seq1, seq2], outputs=pred)
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model
