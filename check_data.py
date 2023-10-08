import common
import random

# Let's visualize some random training examples
random_index = random.randint(0, len(common.train_df)-5) # create random indexes not higher than the total number of samples
for row in common.train_df_shuffeled[["text", "target"]][random_index:random_index+5].itertuples():
  _, text, target = row
  print(f"Target: {target}", "(real disaster)" if target > 0 else "(not real disaster)")
  print(f"Text:\n{text}\n")
  print("---\n")

# Choose a random sentence from the training dataset and tokenize it
random_sentence = random.choice(common.train_sentences)
print(f"Original text:\n{random_sentence}\
      \n\nVectorized version:")
common.text_vectorizer([random_sentence])

 #Get the unique words in the vocabulary
words_in_vocab = common.text_vectorizer.get_vocabulary()
top_5_words = words_in_vocab[:5] # most common tokens (notice the [UNK] token for "unknown" words)
bottom_5_words = words_in_vocab[-5:] # least common tokens
print(f"Number of words in vocab: {len(words_in_vocab)}")
print(f"Top 5 most common words: {top_5_words}")
print(f"Bottom 5 least common words: {bottom_5_words}")