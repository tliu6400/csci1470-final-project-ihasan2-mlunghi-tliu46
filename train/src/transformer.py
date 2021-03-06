import numpy as np
import tensorflow as tf

class Four_Headed_Attention(tf.keras.layers.Layer):

    def __init__(self, emb_sz, use_mask):
        super(Four_Headed_Attention, self).__init__()

        # Hyperparameters
        self.emb_sz = emb_sz
        self.num_heads = 4
        self.head_dim = self.emb_sz // 4
        self.use_mask = use_mask

        # Keys matrices
        self.K = [tf.keras.layers.Dense(self.head_dim, kernel_initializer="glorot_uniform") for _ in range(self.num_heads)]
        # Values matrices
        self.V = [tf.keras.layers.Dense(self.head_dim, kernel_initializer="glorot_uniform") for _ in range(self.num_heads)]
        # Queries matrices
        self.Q = [tf.keras.layers.Dense(self.head_dim, kernel_initializer="glorot_uniform") for _ in range(self.num_heads)]
        # Dense layer
        self.w = tf.keras.layers.Dense(self.emb_sz)

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        total_attention = []
        # Compute attention for each head
        for i in range(self.num_heads):
            attention = self.__attention_matrix(self.K[i](inputs_for_keys), self.Q[i](inputs_for_queries), self.use_mask)
            total_attention.append(tf.matmul(attention, self.V[i](inputs_for_values)))
        total_attention = tf.concat(total_attention, axis=2)
        return self.w(total_attention)

    def __attention_matrix(self, K, Q, use_mask):
        window_size_queries = Q.get_shape()[1]
        window_size_keys = K.get_shape()[1]

        # Compute the mask to ensure only the right parts of a sentence are looked at
        mask = tf.convert_to_tensor(value=np.transpose(np.tril(np.ones((window_size_queries,window_size_keys))*np.NINF,-1),(1,0)),dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask,[-1,window_size_queries,window_size_keys]),[tf.shape(input=K)[0],1,1])

        matrix = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        matrix /= K.get_shape()[2]

        if use_mask:
            matrix += atten_mask
        return tf.nn.softmax(matrix)

class Transformer_Block(tf.keras.layers.Layer):

    def __init__(self, emb_sz, hidden_sz, is_decoder):
        super(Transformer_Block, self).__init__()
        # Intializes the transformer block
        self.self_attention = Four_Headed_Attention(emb_sz, is_decoder)
        self.dropout_self_attention = tf.keras.layers.Dropout(0.3)
        self.is_decoder = is_decoder
        if self.is_decoder:
            self.self_context_attention = Four_Headed_Attention(emb_sz, False)
            self.dropout_context_attention = tf.keras.layers.Dropout(0.3)
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.dense1 = tf.keras.layers.Dense(hidden_sz, activation="relu")
        self.dense2 = tf.keras.layers.Dense(emb_sz)
        self.dropout_dense = tf.keras.layers.Dropout(0.3)

    @tf.function
    def call(self, inputs, context=None):
        attention_out = self.dropout_self_attention(self.self_attention(inputs, inputs, inputs))
        attention_out += inputs
        attention_normalized = self.layer_norm(attention_out)
        if self.is_decoder:
            assert context is not None, "Decoder blocks require context"
            context_attention_out = self.dropout_context_attention(self.self_context_attention(context, context, attention_normalized))
            context_attention_out += attention_normalized
            attention_normalized = self.layer_norm(context_attention_out)
        feed_forward_out = self.dropout_dense(self.dense2(self.dense1(attention_normalized)))
        feed_forward_out += attention_normalized
        feed_forward_out = self.layer_norm(feed_forward_out)
        return tf.nn.relu(feed_forward_out)

class Position_Encoding_Layer(tf.keras.layers.Layer):

    def __init__(self, window_sz, emb_sz):
        super(Position_Encoding_Layer, self).__init__()
        self.positional_embeddings = self.add_weight("pos_embed", shape=[window_sz, emb_sz])

    @tf.function
    def call(self, x):
        return x + self.positional_embeddings

class Transformer(tf.keras.Model):

    def __init__(self, vocab, reverse_vocab):
        super(Transformer, self).__init__()

        # Hyperparameters
        self.batch_sz = 128
        self.num_layers = 4
        self.num_heads = 4
        self.emb_sz = 512
        self.hidden_sz = 512
        self.vocab = vocab
        self.reverse_vocab = reverse_vocab
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        self.vocab_sz = len(vocab)

        # Embedding matrices
        self.embedding_matrix = tf.Variable(tf.random.normal([self.vocab_sz, self.emb_sz], stddev=.1))
        self.positional_encoding_encoder = Position_Encoding_Layer(33, self.emb_sz)
        self.positional_encoding_decoder = Position_Encoding_Layer(33, self.emb_sz)

        # Encoder layers
        self.encoder1 = Transformer_Block(self.emb_sz, self.hidden_sz, False)
        self.encoder2 = Transformer_Block(self.emb_sz, self.hidden_sz, False)
        self.encoder3 = Transformer_Block(self.emb_sz, self.hidden_sz, False)
        self.encoder4 = Transformer_Block(self.emb_sz, self.hidden_sz, False)

        # Decoder layers
        self.decoder1 = Transformer_Block(self.emb_sz, self.hidden_sz, True)
        self.decoder2 = Transformer_Block(self.emb_sz, self.hidden_sz, True)
        self.decoder3 = Transformer_Block(self.emb_sz, self.hidden_sz, True)
        self.decoder4 = Transformer_Block(self.emb_sz, self.hidden_sz, True)

        # Dense layers
        self.dense1 = tf.keras.layers.Dense(self.hidden_sz, activation="relu")
        self.dense2 = tf.keras.layers.Dense(self.hidden_sz, activation="relu")
        self.dense3 = tf.keras.layers.Dense(self.vocab_sz, activation="softmax")

    @tf.function
    def call(self, encoder_input, decoder_input):
        # Pass input through encoder layers
        encoder_embedding = tf.nn.embedding_lookup(self.embedding_matrix, encoder_input)
        encoder_embedding = self.positional_encoding_encoder(encoder_embedding)

        encoder1_output = self.encoder1(encoder_embedding)
        encoder2_output = self.encoder2(encoder1_output)
        encoder3_output = self.encoder3(encoder2_output)
        encoder4_output = self.encoder4(encoder3_output)

        # Pass input through decoder layers
        decoder_embedding = tf.nn.embedding_lookup(self.embedding_matrix, decoder_input)
        decoder_embedding = self.positional_encoding_decoder(decoder_embedding)

        decoder1_output = self.decoder1(decoder_embedding, context=encoder4_output)
        decoder2_output = self.decoder2(decoder1_output, context=encoder4_output)
        decoder3_output = self.decoder3(decoder2_output, context=encoder4_output)
        decoder4_output = self.decoder4(decoder3_output, context=encoder4_output)

        # Pass decoder output through dense layers
        dense1_output = self.dense1(decoder4_output)
        dense2_output = self.dense2(dense1_output)
        probs = self.dense3(dense2_output)

        return probs

    def loss(self, probs, labels, mask):
        probs = tf.boolean_mask(probs, mask)
        labels = tf.boolean_mask(labels, mask)
        # Compute sparse categorical cross-entropy loss
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs, from_logits=False))
