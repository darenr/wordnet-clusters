# Tag Clustering using `wordnet` and `word2vec` distance metrics

Clustering a set of `wordnet` synsets using `k-means`, the `wordnet` pair-wise distance (semantic relatedness) of word senses using the [Edge Counting method of the of Wu & Palmer (1994)](https://pdfs.semanticscholar.org/6eff/221e1cf5ae28ce7dcb60515d028b98e37aa5.pdf) is mapped to the euclidean distance to allow K-means to converge preserving the original pair-wise relationship.

By toggling `use_wordnet = False` to `True` the distance metric between words will use a `GloVe` model `glove.6B.300d_word2vec.txt` (this must be in the [word2vec format](https://radimrehurek.com/gensim/scripts/glove2word2vec.html)) and the `word2vec` similarity value

`extras` folder is proof of concept/experimentations

# To Use:

- create a newline delimited file with a list of `wordnet` senses (eg. data/example_tags.txt)
- to use wordnet set `use_wordnet=True`, to use `word2vec` `use_wordnet=False`
- ```python generate-tag-clusters.py data/example_tags.txt 25 0.7```
  - 25 is the number of clusters to segment the list of `wordnet` senses into.
  - 0.7 is the similarity threshold, below this the words are considered not similar
- results places into the `results` folder as a json file
