# imports
import csv
import json
import numpy as np
import re


# for getting word to index mapping (formed using IMDB data)
def get_word_index(path='imdb_word_index.json'):
    """
    Gives word to index mapping

    # Arguments
        path: where the data is saved
    # Returns
        The word_to_index dictionary
    """
    with open(path) as f:
        return json.load(f)

# word to index mapping
word_to_index = get_word_index()
# index to word mapping
index_to_word = {k: i for i, k in word_to_index.items()}

# open the csv file, preprocess the data and save as twitter.npz for later use
with open('twitter_dataset.csv', 'r', encoding="utf8") as csv_file:
    reader = csv.reader(csv_file)
    # skip headers
    next(reader, None)
    # lists for storing tweets and labels after preprocessing
    tweets = []
    labels = []
    i = 1
    for row in reader:
        label = row[0]
        tweet = row[1]
        if(i % 1000 == 0):
            print("processed {} examples.".format(i))

        # remove urls
        tweet = re.sub(r'\b(?:(?:https?|ftp)://)?\w[\w-]*(?:\.[\w-]+)+\S*',
                       ' ', tweet.lower())
        # remove mentions
        tweet = re.sub(r'(^|[^@\w])@(\w{1,15})\b', ' ', tweet)
        # remove every characters other than a-z, 0-9 and apostrophe
        tweet = re.sub(r'[^\w\']|_', ' ', tweet)
        tweet = tweet.strip()
        # remove multiple white spaces
        tweet = re.sub(r'(\s)+', ' ', tweet)
        # split each sentences to list of words
        tweet = tweet.split()
        # map from words to index (if word not present put -1)
        for index, word in enumerate(tweet):
            tweet[index] = word_to_index.get(word, -1)
        # store the tweet only if 5 or more words present
        if(len(tweet) > 4):
            tweets.append(tweet)
            labels.append(label)
        i += 1

# make 1D numpy arrays for tweets and labels
x_test = np.array(tweets)
y_test = np.array(labels)

# store them as twitter.npz
np.savez_compressed('twitter.npz', x_test=x_test, y_test=y_test)
