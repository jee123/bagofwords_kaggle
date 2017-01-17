from bs4 import BeautifulSoup
import pandas as pd
import re
from nltk.corpus import stopwords
import sys
import os
import glob
from nltk.stem.porter import *

def strip_newsgroup_header(text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.
    """
    _before, _blankline, after = text.partition('\n\n')
    return after


_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')


def strip_newsgroup_quoting(text):
    """
    Given text in "news" format, strip lines beginning with the quote
    characters > or |, plus lines that often introduce a quoted section
    (for example, because they contain the string 'writes:'.)
    """
    good_lines = [line for line in text.split('\n')
                  if not _QUOTE_RE.search(line)]
    return '\n'.join(good_lines)


def strip_newsgroup_footer(text):
    """
    Given text in "news" format, attempt to remove a signature block.

    As a rough heuristic, we assume that signatures are set apart by either
    a blank line or a line made of hyphens, and that it is the last such line
    in the file (disregarding blank lines at the end).
    """
    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip('-') == '':
            break

    if line_num > 0:
        return '\n'.join(lines[:line_num])
    else:
        return text

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

stemmer = PorterStemmer()


def review_to_wordlist( review, remove_stopwords=False ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #   
        # 1. Remove HTML
        review_text = BeautifulSoup(review,"html.parser").get_text()
        #   
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #   
        # 3. Convert words to lower case and split them
        words = review_text.lower().split()
        #   
        # 4. Optionally remove stop words (false by default)
        if remove_stopwords:
                stops = set(stopwords.words("english"))
                words = [w for w in words if not w in stops]
        #   
        # 5. Return a list of words
        return(words)


import nltk.data
#nltk.download()

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
                # If a sentence is empty, skip it
                if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                	sentences.append( review_to_wordlist( raw_sentence,remove_stopwords ))
        #
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences



#Now apply this function to prepare our data for input to Word2Vec
sentences = []  # Initialize an empty list of sentences

#file_name = sys.argv[1]
file_name = "/home/tushar/topic_modeling/Users/tjee/Downloads/word2vec_kaggle/gameon/8514"
#with open(file_name, 'r') as fp:
#	lines = fp.read()
#	lines = strip_newsgroup_footer(lines)
#	lines = strip_newsgroup_header(lines)
#	lines = strip_newsgroup_quoting(lines)
#	lines = lines.encode("utf8").strip()

import codecs
with codecs.open(file_name, mode= 'rb', encoding='utf-8') as fp:
	lines = fp.read()
	lines = strip_newsgroup_footer(lines)
	lines = strip_newsgroup_header(lines)
	lines = strip_newsgroup_quoting(lines)
#for review in lines:
sentences += review_to_sentences(lines.decode("utf8"), tokenizer)

#Check how many sentences we have in total
print "sentence size is %d " %(len(sentences))

#print sentences

import csv
#with open("/home/tushar/topic_modeling/Users/tjee/Downloads/word2vec_kaggle/output_bow.csv", "wb") as f:
with open("/home/tushar/topic_modeling/Users/tjee/Downloads/word2vec_kaggle/output_bow_trial1.csv", "wb") as f:
	writer = csv.writer(f)
	writer.writerows(sentences) 





