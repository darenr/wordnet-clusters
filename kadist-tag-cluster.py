#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import cluster, datasets
import numpy as np
import json
import sys
import codecs
from nltk.corpus import wordnet
from collections import defaultdict


def mk_synset(w):
  word = w.strip()
  if '.' in word:
    return wordnet.synset(word)
  else:
    print 'Error, invalid synset name', w
    sys.exit(-1)

def load_kadist_tags():
  with codecs.open('data/tags.txt', 'rb', 'utf-8') as tagfile:
    return [mk_synset(w) for w in tagfile.readlines()]

def w2w(w1, w2):
  if w1 == w2:
    return 1.0
  else:
    distance = w1.wup_similarity(w2)
    return distance if distance else 0


def make_data_using_wordnet(words):
  list_of_vectors = []
  for word_x in words:
    wordvector = []
    for word_y in words:
      wordvector.append(w2w(word_x, word_y))
    list_of_vectors.append(wordvector)
  data = np.array(np.array(list_of_vectors))
  labels = words
  return (data, labels)


def word_cluster(data, labels, k):
  k_means = cluster.KMeans(n_clusters=2)
  k_means.fit(data)
  for i, label in enumerate(labels):
    print label, k_means.labels_[i]

  d = defaultdict(list)
  for c, l in zip(k_means.labels_, labels):
    d['cluster' + str(c)].append(l.name())
  with codecs.open('data/clusters.json', 'wb', 'utf-8') as outfile:
    outfile.write(json.dumps(d, indent=True))

if __name__ == "__main__":
  print ' *', 'loading tag set...'
  words = load_kadist_tags()
  print ' *', 'generating dataset...'
  data, labels = make_data_using_wordnet(words)
  print ' *', 'clustering...'
  word_cluster(data, labels, k=100)
