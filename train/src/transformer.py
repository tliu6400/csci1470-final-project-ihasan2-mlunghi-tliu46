from math import pi, sqrt
from noisy import noisy
import tensorflow as tf 

def GlorotLinear(input_dim, output_dim):
    initializer = tf.keras.initializers.GlorotUniform()
    return tf.keras.layers.Dense(output_dim, kernel_initializer=initializer)

class MultiHeadAttention(tf.keras.Model):

    def __init__(self, emb_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_head
        if emb_dim % num_heads != 0:
            raise ValueError("MultiHeadAttention Error: emb_dim must be a multiple of num_heads")
        self.K = GlorotLinear(self.emb_dim, self.emb_dim)
        self.V = GlorotLinear(self.emb_dim, self.emb_dim)
        self.Q = GlorotLinear(self.emb_dim, self.emb_dim)
        self.output = GlorotLinear(self.emb_dim, self.emb_dim)

	@tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        pass