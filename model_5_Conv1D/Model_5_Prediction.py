from Model_5_Conv1D import model_5, val_sentences, val_labels
from Calculate_Results import calculate_results
import tensorflow as tf

# Make predictions with model_5
model_5_pred_probs = model_5.predict(val_sentences)
print(model_5_pred_probs[:10])

# Convert model_5 prediction probabilities to labels
model_5_preds = tf.squeeze(tf.round(model_5_pred_probs))
print(model_5_preds[:10])

# Calculate model_5 evaluation metrics 
model_5_results = calculate_results(y_true=val_labels, 
                                    y_pred=model_5_preds)
print(model_5_results)
