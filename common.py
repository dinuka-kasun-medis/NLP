# Download helper functions script
#wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py

from helper_functions import unzip_data, create_tensorboard_callback, plot_loss_curves, compare_historys

# Download data (same as from Kaggle)
#wget "https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip"

# Unzip data
#unzip_data("nlp_getting_started.zip")

import pandas as pd
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
train_df.head()

train_df_shuffeled = train_df.sample(frac=1, random_state = 42)
train_df_shuffeled.head()

test_df.head()

train_df.target.value_counts()

train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffeled["text"].to_numpy(),
                                                                            train_df_shuffeled["target"].to_numpy(),
                                                                            test_size = 0.1,
                                                                            random_state = 42)



text_vetorizer = TextVectorization(max_tokens=None, # how many words in the vocabulary (all of the different words in your text)
                                    standardize="lower_and_strip_punctuation", # how to process text
                                    split="whitespace", # how to split tokens
                                    ngrams=None, # create groups of n-words?
                                    output_mode="int", # how to map tokens to numbers
                                    output_sequence_length=None) # how long should the output sequence of tokens be?
                                    # pad_to_max_tokens=True) # Not valid if using max_tokens=None

# Setup text vectorization with custom variables
max_vocab_length = 10000 # max number of words to have in our vocabulary
max_length = 15 # max length our sequences will be (e.g. how many words from a Tweet does our model see?)

text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)

# Fit the text vectorizer to the training text
text_vectorizer.adapt(train_sentences)

tf.random.set_seed(42)
from tensorflow.keras import layers

embedding = layers.Embedding(input_dim=max_vocab_length, # set input shape
                             output_dim=128, # set size of embedding vector
                             embeddings_initializer="uniform", # default, intialize randomly
                             input_length=max_length, # how long is each input
                             name="embedding_1")

# embedding

# Get a random sentence from training set
random_sentence = random.choice(train_sentences)
print(f"Original text:\n{random_sentence}\
      \n\nEmbedded version:")

# Embed the random sentence (turn it into numerical representation)
sample_embed = embedding(text_vectorizer([random_sentence]))
sample_embed