#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import cluster, datasets
import numpy as np
import json
import sys
import codecs
from nltk.corpus import wordnet
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from gensim.models import KeyedVectors

mpl.rcParams['toolbar'] = 'None'

wvmodel = None

# make False to switch to wordvectors
use_wordnet = False

modelFile = os.environ['HOME'] + "/models/" + "glove.6B.300d_word2vec.txt"

cache = {}


def _mk_synset(w):
    #
    # turn cat.n.01 into the Synset object form
    #
    word = w.strip()
    if '.' in word:
        return wordnet.synset(word)
    else:
        print ' * Error, invalid synset name', w, 'skipping...'
        return None

def _mk_wv_word(s):
    #
    # turn wordnet Synset into word2vec word form
    #   e.g. cat.n.01 -> 'cat'
    #   e.g. free_trade.n.01 -> free-trade
    return s.lemmas()[0].name().replace('_', '-')



def load_tags(filename):
    with codecs.open(filename, 'rb', 'utf-8') as tagfile:
        lines = [line for line in tagfile.readlines() if not line.startswith('#')]
        data = [_mk_synset(w) for w in lines if _mk_synset(w)]
        print ' *', 'loaded', len(data), 'wordnet senses,', len(lines) - len(data), 'rejected'
        return data

#
# wordvectors similarity distance
#

def wv(w1, w2, t):
    # lazy load the wordvector model...
    global wvmodel
    if wvmodel == None:
        print ' *', 'loading wordvector model (', modelFile, ')...'
        wvmodel = KeyedVectors.load_word2vec_format(modelFile, binary=False)
        wvmodel.init_sims(replace=True)  # no more updates, prune memory

    try:
        #
        # since we've got wordnet synset objects (like cat.n.01), we
        # must turn this back into a regular word ('cat') because the
        # word vector GloVe models are plain words with spaces turned
        # into hyphens on phrases (e.g. climate-change, black-and-white)
        #
        wv_w1, wv_w2 = _mk_wv_word(w1), _mk_wv_word(w2)
        distance = wvmodel.similarity(wv_w1, wv_w2)
        return distance if abs(distance) >= t else 0
    except:
        return 0

#
# wordnet wup similarity distance
#


def wup(w1, w2, t):
    distance = w1.wup_similarity(w2)
    if distance:
        if distance >= t:
            return distance
    return 0

#
# wordnet path similarity distancewv
#


def path(w1, w2, t):
    distance = w1.path_similarity(w2)
    if distance:
        if distance >= t:
            return distance
    return 0

#
# Normalized distance between any two words as represented
# by wordnet synsets
#


def word_to_word_distance(w1, w2, t):
    if w1 == w2:
        return 1.0
    else:
        global cache
        s = sorted([w1, w2])
        x = (s[0], s[1])
        if x in cache:
            return cache[x]
        else:
            distances = []
            if use_wordnet:
                distances.append(wup(x[0], x[1], t))
                distances.append(path(x[0], x[1], t))
            else:
                # scale threshold between wm and wv
                distances.append(wv(w1, w2, t / 2.5))
            d = sum(distances) / len(distances)
            cache[x] = d
            return d


def make_data_matrix(words, t):
    list_of_vectors = []
    for word_x in words:
        wordvector = []
        for word_y in words:
            wordvector.append(word_to_word_distance(word_x, word_y, t))
        list_of_vectors.append(wordvector)

    data = np.array(np.array(list_of_vectors))
    labels = words
    return (data, labels)


def show_histogram(d):
    c = {k: len(d[k]) for k in d.keys()}
    bars, heights = zip(*c.items())
    y_pos = range(len(bars))
    plt.bar(y_pos, heights)
    plt.xticks(y_pos, bars, rotation=90)
    plt.show()


def word_cluster(data, labels, k, show_histogram_plot=False):
    k_means = cluster.KMeans(n_clusters=k)
    k_means.fit(data)

    # for i, label in enumerate(labels):
    #    print ' *', label, k_means.labels_[i]

    d = defaultdict(list)
    for c, l in zip(k_means.labels_, labels):
        d['cluster' + str(c)].append(l.name())

    fname = 'results/clusters'
    fname += "_wn" if use_wordnet else "_wv"
    fname += '_k' + str(k) + '.json'

    with codecs.open(fname, 'wb', 'utf-8') as outfile:
        outfile.write(json.dumps(d, indent=True))
        print ' * saved results to:', fname
        # create histogram of cluster sizes
        if show_histogram_plot:
            show_histogram(d)


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print ' *', 'usage: <words file> <k, for example 20>, <threshold, eg 0.7>'
        sys.exit(-1)

    f = sys.argv[1]
    k = int(sys.argv[2])
    t = float(sys.argv[3])

    words = load_tags(f)

    print ' *', 'generating dataset...'
    data, labels = make_data_matrix(words, t)

    print ' *', 'clustering...'
    word_cluster(data, labels, k=k, show_histogram_plot=True)
