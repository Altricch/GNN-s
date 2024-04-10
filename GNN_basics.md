### Basics: 
In GNN we have node level predictions, Edge level predictions (also called Link Predictions) and even graph level predictions

#### Difficulty of GNN: 
Size and shape of graphs change and cannot just be resized. Also, two different graphs can be structurally the same, thus they are the same -> Permutation invariance (thus we cannot used Adjacency matrix as input) Last, graph exists in non-euclidean space.

Here, we use representation learning hrough node level emebdding (each node knows something about the other nodes). This is learned through message passing layers

#### Graph Convolution / Message Passing:
Use neighborhood of node and combine info into new embedding. Message passing can happen through various aggregation methods to combine information. Of course (as in CNNs) the convolution can change the size of the embeddings.

Stacking too many message passing layers can lead to oversmoothing (GNNs exponentially loose expressive power for node classifications (source: https://disco.ethz.ch/courses/fs21/seminar/talks/GNN_Oversmoothing.pdf)). This means that we will ultimately. Update functions can be mean, max, NN or RNN's and Aggregate functions can be Mean Max Normalized Sum and NNs.

Advanced methods uses aggregation as a normalized sum of the neighboring states or use MLP to do the aggregate of neighboring states (which then becomes learnable). Lastly, Graph attention networks are important as well.

#### GCNConv & GINConv
GCNConv and GINConv are instances 




