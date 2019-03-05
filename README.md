# Clustering using wordnet and word2vec distance metrics

Clustering a set of wordnet synsets using k-means, the wordnet pair-wise distance (semantic relatedness) of word senses using the [Edge Counting method of the of Wu & Palmer (1994)](https://pdfs.semanticscholar.org/6eff/221e1cf5ae28ce7dcb60515d028b98e37aa5.pdf) is mapped to the euclidean distance to allow kmeans to converge preserving the original pair-wise relationship.

By toggling `use_wordnet = False` to `True` the distance metric between words will use a GloVe model `glove.6B.300d_word2vec.txt` - this must be in the [word2vec format](https://radimrehurek.com/gensim/scripts/glove2word2vec.html)

Experimental folder contains a working implmentation of the majorclust algorithm and explores some of the way in which document that are sometimes tagged can be clustered in meaningful ways.
