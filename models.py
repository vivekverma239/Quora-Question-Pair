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


def get_model(max_length=50,
              max_vocab_size=100000,
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

    # Run through CONV + GAP layers
    conv1a = conv1(emb1)
    glob1a = GlobalMaxPooling1D()(conv1a)
    conv1b = conv1(emb2)
    glob1b = GlobalMaxPooling1D()(conv1b)

    conv2a = conv2(emb1)
    glob2a = GlobalMaxPooling1D()(conv2a)
    conv2b = conv2(emb2)
    glob2b = GlobalMaxPooling1D()(conv2b)

    conv3a = conv3(emb1)
    glob3a = GlobalMaxPooling1D()(conv3a)
    conv3b = conv3(emb2)
    glob3b = GlobalMaxPooling1D()(conv3b)

    conv4a = conv4(emb1)
    glob4a = GlobalMaxPooling1D()(conv4a)
    conv4b = conv4(emb2)
    glob4b = GlobalMaxPooling1D()(conv4b)

    conv5a = conv5(emb1)
    glob5a = GlobalMaxPooling1D()(conv5a)
    conv5b = conv5(emb2)
    glob5b = GlobalMaxPooling1D()(conv5b)

    conv6a = conv6(emb1)
    glob6a = GlobalMaxPooling1D()(conv6a)
    conv6b = conv6(emb2)
    glob6b = GlobalMaxPooling1D()(conv6b)

    mergea = Concatenate()([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])
    mergeb = Concatenate()([glob1b, glob2b, glob3b, glob4b, glob5b, glob6b])

    # We take the explicit absolute difference between the two sentences
    # Furthermore we take the multiply different entries to get a different measure of equalness
    diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4 * 128 + 2*32,))([mergea, mergeb])
    mul = Lambda(lambda x: x[0] * x[1], output_shape=(4 * 128 + 2*32,))([mergea, mergeb])

    merge = Concatenate()([diff, mul])


    # The MLP that determines the outcome
    x = Dropout(0.2)(merge)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    pred = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[seq1, seq2], outputs=pred)

    return model
