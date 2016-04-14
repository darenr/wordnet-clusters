import os
from gensim.models import Word2Vec

print 'loading wv model'
model = Word2Vec.load_word2vec_format(os.environ['HOME'] + "/models/glove.42B.300d.txt", binary=False)
print 'model ready'

w1 = 'nostalgia'
w2 = 'memory'
print w1, w2, 'similarity:', model.similarity(w1,w2)

