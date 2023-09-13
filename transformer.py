#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from tensorflow.keras.initializers import TruncatedNormal
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds


# ### Encoder

# In[4]:


class MultiHeadAttention(layers.Layer):
    def __init__(self, hidden_size, num_heads):

        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.projection_dim = hidden_size // num_heads
        self.Q = layers.Dense(hidden_size)
        self.K = layers.Dense(hidden_size)
        self.V = layers.Dense(hidden_size)
        self.out = layers.Dense(hidden_size)

    def attention(self, query, key, value, mask):
        

            #### complete this part ####
        matmul_qk = tf.matmul(query, key, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, value)
        return output, attention_weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, att_mask):
        query = self.Q(inputs)[0]
        key = self.K(inputs)
        value = self.V(inputs)
        attention, self.att_weights = self.attention(query, key, value, att_mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.out(concat_attention)  
        return output


# #### Feed-Forward Sub-Layer

# Unlike the original transformer, BERT uses "GELU" activation function. In this part you should implement the GELU activation function based on the paper provided to you.

# In[5]:


@tf.function

def GELU(x):
    cdf = 0.5 * (1.0 + tf.tanh((math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


# In[6]:


class FFN(layers.Layer):

  def __init__(self, intermediate_size, hidden_size, drop_rate):

      super(FFN, self).__init__()
      self.intermediate = layers.Dense(intermediate_size, activation=GELU, kernel_initializer=TruncatedNormal(stddev=0.02))
      self.out = layers.Dense(hidden_size, kernel_initializer=TruncatedNormal(stddev=0.02))
      self.drop = layers.Dropout(drop_rate)

  def call(self, inputs):

      self.net = tf.keras.Sequential([self.intermediate, self.out,self.drop])
      
      return self.net(inputs)


# #### Add & Norm

# In this part implement the add & norm blocks

# In[7]:


class AddNorm(layers.Layer):

    def __init__(self, LNepsilon, drop_rate):
    
        super(AddNorm, self).__init__()
        self.LN = layers.LayerNormalization(epsilon=LNepsilon)
        self.dropout = layers.Dropout(drop_rate)

    def call(self, sub_layer_in, sub_layer_out):

          #### complete this part ####
        add = sub_layer_in + sub_layer_out
        return self.LN(add)


# #### Residual connections
# 
# Now put together all parts and build the encoder with the residual connections
# 
# 
# 

# In[25]:


class Encoder(layers.Layer):

  def __init__(self, hidden_size, num_heads, intermediate_size, drop_rate=0.1, LNepsilon=1e-12):
      super(Encoder, self).__init__()
      self.multihead_attention = MultiHeadAttention(hidden_size, num_heads)
      self.add_norm1 = AddNorm(LNepsilon, drop_rate)
      self.feed_forward = FFN(intermediate_size, hidden_size, drop_rate)
      self.add_norm2 = AddNorm(LNepsilon, drop_rate)

  def call(self, inputs):
      multihead_output = self.multihead_attention(inputs, inputs, inputs)
      multihead_output = self.add_norm1(multihead_output)
      addnorm_output = self.add_norm1(inputs, multihead_output)
      feedforward_output = self.feed_forward(addnorm_output)
      feedforward_output = self.add_norm2(feedforward_output)
      return self.add_norm2(addnorm_output, feedforward_output)

  def compute_mask(self, x, mask):
      return mask


# ### BERT

# In[26]:


class BertEmbedding(layers.Layer):

    def __init__(self, vocab_size, maxlen, hidden_size):

      super(BertEmbedding, self).__init__()
      self.TokEmb = layers.Embedding(input_dim=vocab_size, output_dim=hidden_size, mask_zero=True)
      self.PosEmb = tf.Variable(tf.random.truncated_normal(shape=(maxlen, hidden_size), stddev=0.02))
      self.LN = layers.LayerNormalization(epsilon=1e-12)
      self.dropout = layers.Dropout(0.1)

    def call(self, inputs):
      token_embedding = self.TokEmb(inputs)
      position_embedding = self.PosEmb[:tf.shape(token_embedding)[1]]
      embeddings = token_embedding + position_embedding
      embeddings = self.LN(embeddings)
      embeddings = self.dropout(embeddings)
      return embeddings
    def compute_mask(self, x, mask=None):
      m = 1-tf.cast(self.TokEmb.compute_mask(x), tf.float32)
      m = m[:, tf.newaxis, tf.newaxis, :]
      return m


# The "pooler" is the last layer you need to put in place.
# For each input sentence, the pooler changes the hidden states of the last encoder layer (which have the shape [batch size, sequence lenght, hidden size]) into a vector representation (which has the shape [batch size, hidden size]).
# The pooler does this by giving a dense layer the hidden state that goes with the first token, which is a special token at the beginning of each sentence.

# In[27]:


class Pooler(layers.Layer):

    def __init__(self, hidden_size):

      super(Pooler, self).__init__()
      self.dense = layers.Dense(hidden_size, activation='tanh')

    def call(self, encoder_out):
      first_token_tensor = encoder_out[:, 0]
      pooled_output = self.dense(first_token_tensor)
      return pooled_output


# Now you should complete the **create_BERT** function in the cell below. This function gets BERT's hyper-parameters as its inputs and return a BERT model. 
# Note that the returned model must have two outputs (just like the pre-trained BERTs): 
# - The hidden states of the last encoder layer
# - Output of the pooler

# In[29]:


def create_BERT(vocab_size, maxlen, hidden_size, num_layers, num_att_heads, intermediate_size, drop_rate=0.1):

  # Create the embedding layer
  embedding_layer = BertEmbedding(vocab_size, maxlen, hidden_size)

  # Create the encoder layers
  encoder_layers = [Encoder(hidden_size, num_att_heads, intermediate_size, drop_rate) for _ in range(num_layers)]

  # Create the pooler layer
  pooler_layer = Pooler(hidden_size)

  # Define the inputs and outputs of the model
  inputs = tf.keras.Input(shape=(maxlen,), dtype=tf.int32)
  mask = tf.keras.Input(shape=(maxlen, maxlen), dtype=tf.int32)
  embeddings = embedding_layer(inputs)
  encoder_out = embeddings
  for encoder_layer in encoder_layers:
      encoder_out = encoder_layer(encoder_out)
  pooled_output = pooler_layer(encoder_out[:, 0])
  outputs = (encoder_out, pooled_output)

  # Create the model
  model = tf.keras.Model(inputs=[inputs, mask], outputs=outputs)

  return model


  


# We will use the Rotten tomatoes critic reviews dataset for this assignment. The zip file is provided to you. Unzip it and run the cells below to split the dataset in training and test sets and prepare it for feeding to the bert model.

# In[30]:


train_reviews, test_reviews = pd.read_csv('train_reviews.csv').values[:, 1:], pd.read_csv('test_reviews.csv').values[:, 1:]
(train_texts, train_labels), (test_texts, test_labels)  = (train_reviews[:,0],train_reviews[:,1]), (test_reviews[:,0],test_reviews[:,1]) 
train_texts = [s.lower() for s in train_texts]
test_texts = [s.lower() for s in test_texts] 
aprx_vocab_size = 20000
cls_token = '[cls]'
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(corpus_generator=train_texts,
                                                        target_vocab_size=aprx_vocab_size,
                                                        reserved_tokens=[cls_token])                                               


# In the following cell, you need to complete the implementation of the encode_sentence function. This function takes as input a sentence and an integer representing the maximum length of the sentence and returns a list of token ids. To implement this function, follow these steps:
# 
# -Use the trained tokenizer to encode the input sentence and obtain a list of token ids.
# 
# -Pad the token id list with zeros to the maximum length specified.
# 
# -Prepend the id of the special token to the beginning of the token id list.

# In[31]:


from tensorflow.keras.preprocessing.sequence import pad_sequences

def encode_sentence(s, maxlen):

  tokenized = tokenizer.encode(s)
  # Pad the token id list with zeros
  tokenized = pad_sequences([tokenized], maxlen=maxlen, truncating='post', padding='post')

  # Prepend the special token id to the list
  tokenized = [tokenizer.vocab_size] + tokenized[0].tolist()

  return tokenized


  return tok_id_list


# In[23]:


MAXLEN = 32
x_train = np.array([encode_sentence(x, MAXLEN) for x in train_texts], dtype=np.int64)
x_test = np.array([encode_sentence(x, MAXLEN) for x in test_texts], dtype=np.int64)
y_train = train_labels.astype(np.int64)
y_test = test_labels.astype(np.int64)


# Now use the functional api and the **create_BERT** function you implemented earlier to create a classifier for the movie reviews dataset.
# Note that the intermediate layer in the feed-forward sub-layer of the encoders is set to $4\times H$ in the original BERT implementation, where $H$ is the hidden layer size. 

# In[32]:


hidden_size = 768
num_heads = 12
num_layers = 12
vocab_size = tokenizer.vocab_size  
model = create_BERT(vocab_size, MAXLEN, hidden_size, num_layers, num_heads, intermediate_size = hidden_size*4, drop_rate=0.1)


# In[ ]:


model.compile(tf.keras.optimizers.Adam(learning_rate=5e-5), "binary_crossentropy", metrics=["accuracy"])
model.summary()


# In[ ]:


history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=2,
    validation_data=(x_test, y_test)
)


# ### Attention Visualization

# In[ ]:


#@title Run this!
import sys

get_ipython().system('test -d bertviz_repo && echo "FYI: bertviz_repo directory already exists, to pull latest version uncomment this line: !rm -r bertviz_repo"')
# !rm -r bertviz_repo # Uncomment if you need a clean pull from repo
get_ipython().system('test -d bertviz_repo || git clone https://github.com/jessevig/bertviz bertviz_repo')
if not 'bertviz_repo' in sys.path:
  sys.path += ['bertviz_repo']

from bertviz import head_view

def call_html():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        '''))


# In order to use bertviz, we need to obtain the attention weights in the encoders of the BERT model implemented in the previous section. To do this, you need to complete the implementation of the get_att_weights function in the following cell. This function takes as input a model (the trained BERT-based model from the previous section) and a list of tokens (an encoded sentence). Here's what you need to do:
# 
# -Feed the input token list to the model to generate the attention weights for that input.
# 
# -Access the att_weights attribute of the MultiHeadAttention sub-layer of each encoder in the model and add them all to a list.
# 
# -Return the list (which should be a list of Tensors).

# In[9]:


def get_att_weights(model, tok_id_list):
  
#### complete this part ####

  return att_weights


# In[16]:


import torch
def get_att_tok(model, sent):

  maxlen = model.layers[0].input_shape[0][-1]
  encoded_toks = encode_sentence(sent, maxlen)
  att_weights = get_att_weights(model, encoded_toks)
  pad_start_idx = np.min(np.where(np.array(encoded_toks) == 0))
  toks = encoded_toks[:pad_start_idx]
  atts = []
  for att in att_weights:
    layer_att = torch.FloatTensor(att[:, :, :pad_start_idx, :pad_start_idx].numpy())
    atts.append(layer_att)
  toks = [tokenizer.decode([m]) for m in toks]
  return toks, atts


# #### Attention visualization
# now give a sample sentence in the context of giving your opinion about a movie and visualize the attention. for example "I liked that movie"

# In[ ]:


sentence = "Your sentence"
toks, atts = get_att_tok(model, sentence.lower())
call_html()
head_view(atts, toks)

