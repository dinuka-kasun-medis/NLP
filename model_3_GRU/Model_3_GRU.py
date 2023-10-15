#Build an RNN using the GRU cell
import tensorflow as tf
from tensorflow.keras import layers
import common
from helper_functions import create_tensorboard_callback, calculate_results

# Create directory to save TensorBoard logs
SAVE_DIR = "model_logs"

model_3_embedding = layers.Embedding(input_dim=common.max_vocab_length,
                                     output_dim=128,
                                     embeddings_initializer="uniform",
                                     input_length=common.max_length,
                                     name="embedding_3")

# Build an RNN using the GRU cell
inputs = layers.Input(shape=(1,), dtype="string")
x = common.text_vectorizer(inputs)
x = model_3_embedding(x)
# x = layers.GRU(64, return_sequences=True) # stacking recurrent cells requires return_sequences=True
x = layers.GRU(64)(x) 
# x = layers.Dense(64, activation="relu")(x) # optional dense layer after GRU cell
outputs = layers.Dense(1, activation="sigmoid")(x)
model_3 = tf.keras.Model(inputs, outputs, name="model_3_GRU")

# Compile GRU model
model_3.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Get a summary of the GRU model
model_3.summary()

# Fit model
model_3_history = model_3.fit(common.train_sentences,
                             common.train_labels,
                              epochs=5,
                              validation_data=(common.val_sentences, common.val_labels),
                              callbacks=[create_tensorboard_callback(SAVE_DIR, "GRU")])

# Make predictions on the validation data
model_3_pred_probs = model_3.predict(common.val_sentences)
model_3_pred_probs.shape, model_3_pred_probs[:10]

# Convert prediction probabilities to prediction classes
model_3_preds = tf.squeeze(tf.round(model_3_pred_probs))
model_3_preds[:10]

# Calcuate model_3 results
model_3_results = calculate_results(y_true=common.val_labels, 
                                    y_pred=model_3_preds)
model_3_results