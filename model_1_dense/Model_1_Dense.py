# Create tensorboard callback (need to create a new one for each model)
from helper_functions import create_tensorboard_callback

import tensorflow as tf
from tensorflow.keras import layers
import common

# Create directory to save TensorBoard logs
SAVE_DIR = "model_logs"

# Build model with the Functional API
inputs = layers.Input(shape=(1,), dtype="string") # inputs are 1-dimensional strings
x = common.text_vectorizer(inputs) # turn the input text into numbers
x = common.embedding(x) # create an embedding of the numerized numbers
x = layers.GlobalAveragePooling1D()(x) # lower the dimensionality of the embedding (try running the model without this layer and see what happens)
outputs = layers.Dense(1, activation="sigmoid")(x) # create the output layer, want binary outputs so use sigmoid activation
model_1 = tf.keras.Model(inputs, outputs, name="model_1_dense") # construct the model

# Compile model
model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Get a summary of the model
model_1.summary()

# Fit the model
model_1_history = model_1.fit(common.train_sentences, # input sentences can be a list of strings due to text preprocessing layer built-in model
                              common.train_labels,
                              epochs=5,
                              validation_data=(common.val_sentences, common.val_labels),
                              callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR, 
                                                                     experiment_name="simple_dense_model")])

# Check the results
model_1.evaluate(common.val_sentences, common.val_labels)

common.embedding.weights

embed_weights = model_1.get_layer("embedding_1").get_weights()[0]
print(embed_weights.shape)

