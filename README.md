# wordnet-clusters

Clustering a set of wordnet synsets using k-means, the wordnet pair-wise distance (semantic relatedness of word senses using the edge counting method of the of Wu & Palmer (1994) is mapped to the euclidean distance to allow kmeans to converge preserving the original pair-wise relationship.

Experimental folder contains a working implmentation of the majorclust algorithm and explores some of the way in which document that are sometimes tagged can be clustered in meaningful ways.
