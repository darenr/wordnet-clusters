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
import os
from gensim.models import KeyedVectors

wvmodel = None

# make False to switch to wordvectors
use_wordnet = True

modelFile = os.environ['HOME'] + "/models/" + "glove.6B.300d_word2vec.txt"

cache = {}

def mk_synset(w):
    word = w.strip()
    if '.' in word:
        return wordnet.synset(word)
    else:
        print ' * Error, invalid synset name', w, 'skipping...'
        return None


def load_tags(filename):
    with codecs.open(filename, 'rb', 'utf-8') as tagfile:
        data = [mk_synset(w) for w in tagfile.readlines() if mk_synset(w)]
        print ' *', 'loaded', len(data), 'wordnet senses'
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
        distance = wvmodel.similarity(w1.lemmas()[0].name(), w2.lemmas()[0].name())
        print w1.name(), w2.name(), 'distance: ', distance
        return distance if distance >= t else 0
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
    labels, values = zip(*c.items())
    indexes = np.arange(len(labels))
    width = 1
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels, rotation='vertical')
    plt.show()

def word_cluster(data, labels, k, show_histogram_plot=False):
    k_means = cluster.KMeans(n_clusters=k)
    k_means.fit(data)
    for i, label in enumerate(labels):
        print label, k_means.labels_[i]

    d = defaultdict(list)
    for c, l in zip(k_means.labels_, labels):
        d['cluster' + str(c)].append(l.name())

    fname = 'results/clusters'
    fname += "_wn" if use_wordnet else "_wv"
    fname += '_k' + str(k) + '.json'

    with codecs.open(fname, 'wb', 'utf-8') as outfile:
        outfile.write(json.dumps(d, indent=True))
        print ' * saved results to', fname
        # create histogram of cluster sizes
        if show_histogram_plot:
            show_histogram(d)


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print 'usage: <words file> <k, for example 20>, <threshold, eg 0.7>'
        sys.exit(-1)

    f = sys.argv[1]
    k = int(sys.argv[2])
    t = float(sys.argv[3])

    words = load_tags(f)

    print ' *', 'generating dataset...'
    data, labels = make_data_matrix(words, t)

    print ' *', 'clustering...'
    word_cluster(data, labels, k=k, show_histogram_plot=True)
