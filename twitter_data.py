# imports
import numpy as np
import json


def load_data(path='./datasets/twitter.npz', seed=6699, start_char=1,
              index_from=3, maxlen=None, num_words=None,
              oov_char=2, skip_top=0):
    """
    Loads the Twitter dataset

    # Arguments
        path: where the data is saved
        num_words: max number of words to include (words arranged as per
                    the frequency of occurence in IMDB dataset)
        skip_top: skip the top N most frequently occurring words in IMDB data
        maxlen: sequences longer than this will be filtered out
        seed: random seed for sample shuffling
        start_char: The start of a sequence will be marked with
                     this character
        oov_char: words that were cut out will be replaced with this character
        index_from: index actual words with this index and higher
    # Returns
        Tuple of Numpy arrays: `(x_test, y_test)`
    """

    # load the data
    with np.load(path) as f:
        x_test, labels_test = f['x_test'], f['y_test']

    # for shuffle consistency
    np.random.seed(seed)

    # shuffle data
    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    xs = list(x_test)
    labels = list(labels_test)

    # each sequence starts with start_char and ,
    # word indexing starts from index_from onwards
    xs = [[start_char] + [w + index_from for w in x] for x in xs]

    # filter out sequences with length less than maxlen
    if(maxlen):
        xs = [x for x in xs if len(x) < maxlen]
        if not xs:
            raise ValueError('filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', left no sequence. '
                             'Increase maxlen.')
        labels = labels[:len(xs)]

    # if num_words not specified,
    # set num_words to total number of words present as per IMDB data
    if not num_words:
        num_words = max([max(x) for x in xs])

    # put oov_char for words not needed
    if oov_char is not None:
        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x]
              for x in xs]
    # if oov_char is None, skip the words that are not needed
    else:
        xs = [[w for w in x if skip_top <= w < num_words]
              for x in xs]

    # make numpy array for test data and labels
    x_test, y_test = np.array(xs), np.array(labels)

    return (x_test, y_test)


def get_word_index(path='./datasets/imdb_word_index.json'):
    """
    Gives word to index mapping as per IMDB data

    # Arguments
        path: where the data is saved
    # Returns
        The word_to_index dictionary
    """
    with open(path) as f:
        return json.load(f)
