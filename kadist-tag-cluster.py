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

cache = {}

def mk_synset(w):
  word = w.strip()
  if '.' in word:
    return wordnet.synset(word)
  else:
    print ' * Error, invalid synset name', w, 'skipping...'
    return None


def load_kadist_tags():
  with codecs.open('data/tags.txt', 'rb', 'utf-8') as tagfile:
    return [mk_synset(w) for w in tagfile.readlines() if mk_synset(w)]


def wup(w1, w2, t):
  distance = w1.wup_similarity(w2)
  if distance:
    if distance >= t:
      return distance
  return 0


def path(w1, w2, t):
  distance = w1.path_similarity(w2)
  if distance:
    if distance >= t:
      return distance
  return 0


def w2w(w1, w2, t):
  if w1 == w2:
    return 1.0
  else:
    s = sorted([w1,w2])
    x=(s[0], s[1])
    if x in cache:
      return cache[x]
    else:
      distance1 = wup(x[0], x[1], t)
      distance2 = path(x[0], x[1], t)
      d = (distance1 + distance2) / 2.0
      cache[x] = d
      return d


def make_data_using_wordnet(words, t):
  list_of_vectors = []
  for word_x in words:
    wordvector = []
    for word_y in words:
      wordvector.append(w2w(word_x, word_y, t))
    list_of_vectors.append(wordvector)
  data = np.array(np.array(list_of_vectors))
  labels = words
  return (data, labels)

def histogram(d):
  c = {k:len(d[k]) for k in d.keys()}
  labels, values = zip(*c.items())
  indexes = np.arange(len(labels))
  width=1
  plt.bar(indexes, values, width)
  plt.xticks(indexes+width*0.5, labels)
  plt.show()


def word_cluster(data, labels, k):
  k_means = cluster.KMeans(n_clusters=k)
  k_means.fit(data)
  for i, label in enumerate(labels):
    print label, k_means.labels_[i]

  d = defaultdict(list)
  for c, l in zip(k_means.labels_, labels):
    d['cluster' + str(c)].append(l.name())
  fname = 'data/clusters_k' + str(k) + '.json'
  with codecs.open(fname, 'wb', 'utf-8') as outfile:
    outfile.write(json.dumps(d, indent=True))
    print ' * Saved results to', fname
    # create historgram of cluster sizes
    histogram(d)

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print 'usage: <k>, <threshold>'
    sys.exit(-1)

  k = int(sys.argv[1])
  t = float(sys.argv[2])
  print ' *', 'k=', k, 't=', t
  print ' *', 'loading tag set...'
  words = load_kadist_tags()
  print ' *', 'generating dataset...'
  data, labels = make_data_using_wordnet(words, t)
  print ' *', 'clustering...'
  word_cluster(data, labels, k=k)
