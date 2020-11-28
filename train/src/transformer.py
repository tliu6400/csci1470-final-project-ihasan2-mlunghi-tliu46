import tensorflow as tf 

class Four_Headed_Attention(tf.keras.layers.Layer):

    def __init__(self, emb_sz, use_mask):
        super(MultiHeadAttention, self).__init__()

        self.emb_sz = emb_sz
		self.num_heads = 4
		self.head_dim = self.emb_sz // 4
		self.use_mask = use_mask

		self.K1 = self.add_weight(shape=[self.emb_sz, self.head_dim], initializer="glorot_uniform")
		self.V1 = self.add_weight(shape=[self.emb_sz, self.head_dim], initializer="glorot_uniform")
		self.Q1 = self.add_weight(shape=[self.emb_sz, self.head_dim], initializer="glorot_uniform")
		self.K2 = self.add_weight(shape=[self.emb_sz, self.head_dim], initializer="glorot_uniform")
		self.V2 = self.add_weight(shape=[self.emb_sz, self.head_dim], initializer="glorot_uniform")
		self.Q2 = self.add_weight(shape=[self.emb_sz, self.head_dim], initializer="glorot_uniform")
		self.K3 = self.add_weight(shape=[self.emb_sz, self.head_dim], initializer="glorot_uniform")
		self.V3 = self.add_weight(shape=[self.emb_sz, self.head_dim], initializer="glorot_uniform")
		self.Q3 = self.add_weight(shape=[self.emb_sz, self.head_dim], initializer="glorot_uniform")
        self.K4 = self.add_weight(shape=[self.emb_sz, self.head_dim], initializer="glorot_uniform")
		self.V4 = self.add_weight(shape=[self.emb_sz, self.head_dim], initializer="glorot_uniform")
		self.Q4 = self.add_weight(shape=[self.emb_sz, self.head_dim], initializer="glorot_uniform")

        self.w = tf.keras.layers.Dense(self.emb_sz)

	@tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
		K1 = tf.tensordot(inputs_for_keys, self.K1, axes=[2, 0])
		V1 = tf.tensordot(inputs_for_values, self.V1, axes=[2, 0])
		Q1 = tf.tensordot(inputs_for_queries, self.Q1, axes=[2, 0])
		z1 = tf.matmul(self.__attention_matrix(K1, Q1, self.use_mask), V1)
		K2 = tf.tensordot(inputs_for_keys, self.K2, axes=[2, 0])
		V2 = tf.tensordot(inputs_for_values, self.V2, axes=[2, 0])
		Q2 = tf.tensordot(inputs_for_queries, self.Q2, axes=[2, 0])
		z2 = tf.matmul(self.__attention_matrix(K2, Q2, self.use_mask), V2)
		K3 = tf.tensordot(inputs_for_keys, self.K3, axes=[2, 0])
		V3 = tf.tensordot(inputs_for_values, self.V3, axes=[2, 0])
		Q3 = tf.tensordot(inputs_for_queries, self.Q3, axes=[2, 0])
		z3 = tf.matmul(self.__attention_matrix(K3, Q3, self.use_mask), V3)
        K4 = tf.tensordot(inputs_for_keys, self.K4, axes=[2, 0])
		V4 = tf.tensordot(inputs_for_values, self.V4, axes=[2, 0])
		Q4 = tf.tensordot(inputs_for_queries, self.Q4, axes=[2, 0])
		z4 = tf.matmul(self.__attention_matrix(K4, Q4, self.use_mask), V4)

		return self.w(tf.concat([z1, z2, z3], axis=2))

    def __attention_matrix(K, Q, use_mask):
        window_size_queries = Q.get_shape()[1]
        window_size_keys = K.get_shape()[1]

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

		self.self_attention = Four_Headed_Attention(emb_sz, is_decoder)

		self.is_decoder = is_decoder
		if self.is_decoder:
			self.self_context_attention = Multi_Headed(emb_sz, False)

		self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

        self.feed_forward = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_sz, activation='relu'),
                tf.keras.layers.Dense(emb_sz)
            ]
        )

	@tf.function
	def call(self, inputs, context=None):
		attention_out = self.self_attention(inputs, inputs, inputs)
		attention_out += inputs
		attention_normalized = self.layer_norm(attention_out)

		if self.is_decoder:
			context_attention_out = self.self_context_attention(context, context, atten_normalized)
			context_attention_out += attention_normalized
			attention_normalized = self.layer_norm(context_attention_out)

		feed_forward_out = self.feed_forward(attention_normalized)
		feed_forward_out += attention_normalized
		feed_forward_out = self.layer_norm(feed_forward_out)

		return tf.nn.relu(feed_forward_out)

class Transformer(tf.keras.Model):

    def __init__(self, vocab_size):
		super(Transformer, self).__init__()

		self.batch_sz = 8000
        self.num_layers = 4
        self.num_heads = 4
		self.emb_sz = 512
        self.hidden_sz = 512
		self.vocab_sz = vocab_size
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

		self.embedding_matrix = tf.Variable(tf.random.normal([self.vocab_sz, self.emb_sz], stddev=.1))
        self.encoder = tf.keras.Sequential(
            [
                Transformer_Block(self.emb_sz, self.hidden_sz, False),
                Transformer_Block(self.emb_sz, self.hidden_sz, False),
                Transformer_Block(self.emb_sz, self.hidden_sz, False),
                Transformer_Block(self.emb_sz, self.hidden_sz, False)
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                Transformer_Block(self.emb_sz, self.hidden_sz, True)
                Transformer_Block(self.emb_sz, self.hidden_sz, True)
                Transformer_Block(self.emb_sz, self.hidden_sz, True)
                Transformer_Block(self.emb_sz, self.hidden_sz, True)
            ]
        )
        self.feed_forward = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_sz, activation="relu"),
                tf.keras.layers.Dense(self.hidden_sz, activation="relu"),
                tf.keras.layers.Dense(self.vocab_sz, activation="softmax"),
            ]
        )
	
	@tf.function
	def call(self, encoder_input, decoder_input):
		encoder_embedding = tf.nn.embedding_lookup(self.embedding_matrix, encoder_input)
		encoder_output = self.encoder(encoder_embedding)
        decoder_embedding = tf.nn.embedding_lookup(self.embedding_matrix, decoder_input)
		decoder_output = self.decoder(decoder_embedding, context=encoder_output)
        probs = self.feed_forward(decoder_output)
		return probs

	def loss(self, probs, labels, mask):
		probs = tf.boolean_mask(probs, mask)
		labels = tf.boolean_mask(labels, mask)
		return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs, from_logits=False))	