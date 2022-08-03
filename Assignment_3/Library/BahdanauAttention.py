"""
Attention Block

Reference:
https://www.tensorflow.org/text/tutorials/nmt_with_attention
"""

# Imports
import tensorflow as tf

# Main Functions
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        # For Eqn. (4), the    Bahdanau attention
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)

        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query=None, value=None, mask=None):
        # From Eqn. (4), `W1@ht`.
        w1_query = self.W1(query)

        # From Eqn. (4), `W2@hs`.
        w2_key = self.W2(value)

        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask if mask is not None else tf.ones(tf.shape(value)[:-1], dtype=bool)

        context_vector, attention_weights = self.attention(
                inputs = [w1_query, value, w2_key],
                mask=[query_mask, value_mask],
                return_attention_scores = True,
        )

        return context_vector, attention_weights

    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({"units": self.units})
        return config