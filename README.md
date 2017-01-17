# bagofwords_kaggle
topic modelling    
trial 1 is only experimenting on file 8514 in gameon(from 20 Newsgroups Data Set).  
Remaining work left is regarding training word2vec model(trainbagofwords.py) on the remaining files in gameon and finally creating word to vector dictionary.  
        
gameon: contains the 20newsgroup documents
bagofwords2.py : script called in tmp_topic_bow.  
output_bow_trial1.csv : csv containing the processed documents(8514) fed to the word2vec model as a list of lists.  
tmp_topic_bow: script to clean each document inside gameon.  
trainbagofwords.py: training the Word2Vec model.  
