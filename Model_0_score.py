import common
import Model_0_NaiveBayes

baseline_score = Model_0_NaiveBayes.model_0.score(common.val_sentences, common.val_labels)
print(f"Our baseline model achieves an accuracy of: {baseline_score*100:.2f}%")

# Make predictions
baseline_preds = Model_0_NaiveBayes.model_0.predict(common.val_sentences)
baseline_preds[:20]

# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
  -----
  y_true = true labels in the form of a 1D array
  y_pred = predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted" average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results

# Get baseline results
baseline_results = calculate_results(y_true=common.val_labels,
                                     y_pred=baseline_preds)
baseline_results