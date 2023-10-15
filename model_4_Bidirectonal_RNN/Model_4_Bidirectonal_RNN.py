#A standard RNN will process a sequence from left to right, where as a bidirectional RNN will process the sequence from left to right and then again from right to left.
import tensorflow as tf
# Set random seed and create embedding layer (new embedding layer for each model)
tf.random.set_seed(42)
from tensorflow.keras import layers
import common
from helper_functions import create_tensorboard_callback, calculate_results

# Create directory to save TensorBoard logs
SAVE_DIR = "model_logs"

model_4_embedding = layers.Embedding(input_dim=common.max_vocab_length,
                                     output_dim=128,
                                     embeddings_initializer="uniform",
                                     input_length=common.max_length,
                                     name="embedding_4")

# Build a Bidirectional RNN in TensorFlow
inputs = layers.Input(shape=(1,), dtype="string")
x = common.text_vectorizer(inputs)
x = model_4_embedding(x)
# x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x) # stacking RNN layers requires return_sequences=True
x = layers.Bidirectional(layers.LSTM(64))(x) # bidirectional goes both ways so has double the parameters of a regular LSTM layer
outputs = layers.Dense(1, activation="sigmoid")(x)
model_4 = tf.keras.Model(inputs, outputs, name="model_4_Bidirectional")

# Compile
model_4.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Get a summary of our bidirectional model
model_4.summary()

# Fit the model (takes longer because of the bidirectional layers)
model_4_history = model_4.fit(common.train_sentences,
                              common.train_labels,
                              epochs=5,
                              validation_data=(common.val_sentences, common.val_labels),
                              callbacks=[create_tensorboard_callback(SAVE_DIR, "bidirectional_RNN")])

# Make predictions with bidirectional RNN on the validation data
model_4_pred_probs = model_4.predict(common.val_sentences)
model_4_pred_probs[:10]

# Convert prediction probabilities to labels
model_4_preds = tf.squeeze(tf.round(model_4_pred_probs))
model_4_preds[:10]

# Calculate bidirectional RNN model results
model_4_results = calculate_results(common.val_labels, model_4_preds)
model_4_results



