'''
Our model is going to take on a very similar structure to model_1:
Input (text) -> Tokenize -> Embedding -> Layers -> Output (label probability)
'''
import tensorflow as tf
from tensorflow.keras import layers
import common
from helper_functions import create_tensorboard_callback, calculate_results

# Create directory to save TensorBoard logs
SAVE_DIR = "model_logs"

# Set random seed and create embedding layer (new embedding layer for each model)
tf.random.set_seed(42)


model_2_embedding = layers.Embedding(input_dim=common.max_vocab_length,
                                     output_dim=128,
                                     embeddings_initializer="uniform",
                                     input_length=common.max_length,
                                     name="embedding_2")


# Create LSTM model
inputs = layers.Input(shape=(1,), dtype="string")
x = common.text_vectorizer(inputs)
x = model_2_embedding(x)
print(x.shape)
# x = layers.LSTM(64, return_sequences=True)(x) # return vector for each word in the Tweet (you can stack RNN cells as long as return_sequences=True)
x = layers.LSTM(64)(x) # return vector for whole sequence
print(x.shape)
# x = layers.Dense(64, activation="relu")(x) # optional dense layer on top of output of LSTM cell
outputs = layers.Dense(1, activation="sigmoid")(x)
model_2 = tf.keras.Model(inputs, outputs, name="model_2_LSTM")


# Compile model
model_2.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_2.summary()


# Fit model
model_2_history = model_2.fit(common.train_sentences,
                              common.train_labels,
                              epochs=5,
                              validation_data=(common.val_sentences, common.val_labels),
                              callbacks=[create_tensorboard_callback(SAVE_DIR, 
                                                                     "LSTM")])

# Make predictions on the validation dataset
model_2_pred_probs = model_2.predict(common.val_sentences)
model_2_pred_probs.shape, model_2_pred_probs[:10] # view the first 10

# Round out predictions and reduce to 1-dimensional array
model_2_preds = tf.squeeze(tf.round(model_2_pred_probs))
model_2_preds[:10]

# Calculate LSTM model results
model_2_results = calculate_results(y_true=common.val_labels,
                                    y_pred=model_2_preds)
model_2_results