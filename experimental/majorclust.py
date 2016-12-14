#!/usr/bin/python2.7 -S
# -*- coding: utf-8 -*-

# details: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.85.3073&rep=rep1&type=pdf

import sys
from math import log, sqrt
from itertools import combinations
import codecs
import json
import sys
import math
import os
import collections

def cosine_distance(a, b):
    cos = 0.0
    a_tfidf = a["tfidf"]
    for token, tfidf in b["tfidf"].iteritems():
        if token in a_tfidf:
            cos += tfidf * a_tfidf[token]
    return cos

def tag_distance(a,b):
    # simply return a count of the number of tokens in common
    sim = len([x for x in a['tokens'] if x in set(b['tokens'])])
    return sim

def distance_function(a,b):
    #
    # autoselect distance metrics, for documents we break them into tokens and use
    # cosine_distance, but for tags which shouldn't be tokenized the distance
    # between any two sets of "tags"
    #
    if len(a['tokens']) < 10 or len(b['tokens']) < 5:
        return tag_distance(a,b)
    else:
        return cosine_distance(a,b)


def normalize(features):
    norm = 1.0 / sqrt(sum(i**2 for i in features.itervalues()))
    for k, v in features.iteritems():
        features[k] = v * norm
    return features


def add_tfidf_to(documents):
    tokens = {}
    for doc in documents:
        id = doc['id']
        tf = {}
        doc["tfidf"] = {}
        doc_tokens = doc.get("tokens", [])
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1
        num_tokens = len(doc_tokens)
        if num_tokens > 0:
            for token, freq in tf.iteritems():
                tokens.setdefault(token, []).append((id, float(freq) / num_tokens))

    doc_count = float(len(documents))
    for token, docs in tokens.iteritems():
        idf = log(doc_count / len(docs))
        for id, tf in docs:
            tfidf = tf * idf
            if tfidf > 0:
                documents[id]["tfidf"][token] = tfidf

    for doc in documents:
        doc["tfidf"] = normalize(doc["tfidf"])


def choose_cluster(node, cluster_lookup, edges):
    new = cluster_lookup[node]
    if node in edges:
        seen, num_seen = {}, {}
        for target, weight in edges.get(node, []):
            seen[cluster_lookup[target]] = seen.get(
                cluster_lookup[target], 0.0) + weight
        for k, v in seen.iteritems():
            num_seen.setdefault(v, []).append(k)
        new = num_seen[max(num_seen)][0]
    return new


def majorclust(graph):
    cluster_lookup = dict((node, i) for i, node in enumerate(graph.nodes))

    count = 0
    movements = set()
    finished = False
    while not finished:
        finished = True
        for node in graph.nodes:
            new = choose_cluster(node, cluster_lookup, graph.edges)
            move = (node, cluster_lookup[node], new)
            if new != cluster_lookup[node] and move not in movements:
                movements.add(move)
                cluster_lookup[node] = new
                finished = False

    clusters = {}
    for k, v in cluster_lookup.iteritems():
        clusters.setdefault(v, []).append(k)

    return clusters.values()


def get_distance_graph(documents):
    class Graph(object):

        def __init__(self):
            self.edges = {}

        def add_edge(self, n1, n2, w):
            self.edges.setdefault(n1, []).append((n2, w))
            self.edges.setdefault(n2, []).append((n1, w))

    graph = Graph()
    doc_ids = range(len(documents))
    graph.nodes = set(doc_ids)
    for a, b in combinations(doc_ids, 2):
        graph.add_edge(a, b, distance_function(documents[a], documents[b]))
    return graph


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def process_document(d, id, fields):
    tokens = []
    for field in fields:
        if field in d and d[field]:
            if isinstance(d[field], list):
                tokens += d[field]
            elif isinstance(d[field], basestring):
                if len(d[field]) < 30:
                    tokens += [d[field]]
                else:
                    tokens += d[field].split()
            elif isinstance(d[field], int):
                tokens += [str(d[field])]

    if tokens:
        return {
            "id": id,
            "tokens": tokens
        }

    return None

def get_arpedia_documents(enriched_files_folder, fields):
    docs = []
    m = {}

    doc_id = 0
    for filename in os.listdir(enriched_files_folder):
        #if doc_id == 100: return m, docs
        if filename.endswith('.json'):
            with codecs.open(os.path.join(enriched_files_folder, filename), 'rb', 'utf-8') as f:
                enriched_record = json.loads(f.read())
                d = flatten(enriched_record)
                doc = process_document(d, doc_id, fields)
                if doc:
                    m[doc_id] = d
                    docs.append(doc)
                    doc_id += 1

    return m, docs


def main(args):
    fields = ['decade_work_created', 'artist_bio.location', 'description', 'artist_description', 'merged_tags']
    m, documents = get_arpedia_documents(args[1], fields)

    add_tfidf_to(documents)
    dist_graph = get_distance_graph(documents)

    result = []

    for cluster in majorclust(dist_graph):
        membership = list(set([m[doc_id]['artist_name'] for doc_id in cluster if 'artist_name' in m[doc_id]]))
        print len(membership)
        result.append({
            "id":  doc_id,
            "membership": membership,
            "size": len(membership)
        })

    filename = 'clusters-%s.json' % ('-'.join(fields))
    with codecs.open(filename, 'wb', 'utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2, encoding='utf8'))
    print 'written results to', filename

if __name__ == '__main__':
    main(sys.argv)
