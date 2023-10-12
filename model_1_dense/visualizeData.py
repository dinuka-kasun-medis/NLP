
import common
# Get the vocabulary from the text vectorization layer
words_in_vocab = common.text_vectorizer.get_vocabulary()
len(words_in_vocab), words_in_vocab[:10]