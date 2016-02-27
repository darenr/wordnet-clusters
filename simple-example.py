# -*- coding: utf-8 -*-

from sklearn import cluster, datasets
import numpy as np
import json
from nltk.corpus import wordnet
from collections import defaultdict

words = [wordnet.synset(u"cat.n.01"), wordnet.synset(u"dog.n.01"),
         wordnet.synset(u"horse.n.01"), wordnet.synset(u"boat.n.01"), wordnet.synset(u"ship.n.01")]


def w2w(w1, w2):
  if w1 == w2:
    return 1
  else:
    return w1.wup_similarity(w2)

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


def make_data_by_hand():
  ''' make an array of word to word path distances, thus:

         cat  dog   horse boat  ship
  cat    1    0.8   0.6   0.1   0.1
  dog    0.8  1     0.7   0.15  0.15
  horse  0.6  0.7   1     0.1   0.1
  boat   0.1  0.15  0.1   1     0.9
  ship   0.1  0.15  0.1   0.9   1

  '''

  cat = np.array([1.0, 0.8, 0.6, 0.1, 0.1])
  dog = np.array([0.8, 1.0, 0.7, 0.15, 0.15])
  horse = np.array([0.6, 0.7, 1.0, 0.1, 0.1])
  boat = np.array([0.1, 0.15, 0.1, 1.0, 0.9])
  ship = np.array([0.1, 0.15, 0.1, 0.9, 1.0])

  data = np.array([cat, dog, horse, boat, ship])
  labels = np.array(['cat', 'dog', 'horse', 'boat', 'ship'])
  return (data, labels)


def word_cluster(data, labels, k):
  k_means = cluster.KMeans(n_clusters=2)
  k_means.fit(data)
  for i, label in enumerate(labels):
    print label, k_means.labels_[i]

  d = defaultdict(list)
  for c, l in zip(k_means.labels_, labels):
    d['cluster'+str(c)].append(l.name())
  print json.dumps(d, indent=True)

if __name__ == "__main__":
  data, labels = make_data_using_wordnet(words)
  word_cluster(data, labels, k=2)
