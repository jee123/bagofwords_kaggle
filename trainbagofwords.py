# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
import logging
import csv
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

# Set values for various parameters
num_features = 200    # Word vector dimensionality
min_word_count = 5   # Minimum word count
num_workers =64       # Number of threads to run in parallel
context = 5          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

from gensim.models import word2vec
#sentences = word2vec.LineSentence("/home/tushar/topic_modeling/Users/tjee/Downloads/word2vec_kaggle/out_topic_sentences.txt")
#from pathlib2 import Path
#sentences = Path("/home/tushar/topic_modeling/Users/tjee/Downloads/word2vec_kaggle/out_topic_sentences.txt").read_text()
sentences = []
#with open("/home/tushar/topic_modeling/Users/tjee/Downloads/word2vec_kaggle/output_bow.csv", "rb") as fr:
with open("/home/tushar/topic_modeling/Users/tjee/Downloads/word2vec_kaggle/output_bow_trial1.csv", "rb") as fr:
	reader = csv.reader(fr)
	for row in reader:
		sentences.append(row)

#print "size of sentences are : %d" %(len(sentences))
#sentences = "/home/tushar/topic_modeling/Users/tjee/Downloads/word2vec_kaggle/out_topic_sentences.txt"
model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# save the model for later use. You can load it later using Word2Vec.load()
#model_name = "200features_10minwords_10context"
model_name = "200features_5minwords_5context"

model.save(model_name)

#exploring model results
#print model.doesnt_match("man woman child kitchen".split())
#print model.most_similar("man")
#print model.most_similar("awful")
print model.syn0.shape
print "vector coordinates are: \n"
print model.syn0
print "model vocab is as : \n"
# Index2word is a list that contains the names of the words in 
# the model's vocabulary. Convert it to a set, for speed 
#index2word_set = set(model.index2word)
index2word = model.index2word
print index2word
print "model testing and validation : \n"
print model.doesnt_match("Presentations visualization proceedings submission Carderock".split())
#print model.most_similar("submission")

