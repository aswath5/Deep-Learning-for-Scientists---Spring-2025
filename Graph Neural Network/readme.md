# Description

A graph neural network (GNN) is a type of neural network that is specifically designed to work with graph data structures. Graphs are mathematical structures that consist of nodes (also known as vertices) and edges (also known as links) that connect them. GNNs can learn to extract features from graphs and make predictions based on them.

In molecular science, GNNs have become increasingly popular for tasks such as predicting molecular properties, discovering new molecules, and designing new drugs. Molecules can be represented as graphs, where atoms are represented as nodes and bonds between atoms are represented as edges. GNNs can be trained on large datasets of molecular graphs and their associated properties, allowing them to learn to predict properties of new molecules that they have not seen before.

In this Homework, you are required to implement a graph-based neural network model to predict a target property (Regression). The train and test datasets have been provided to you. You will train a GNN on the train dataset, then use it to make predictions on the test dataset. This Homework specifically requires implementation using PyTorch Geometric (PyG): https://pytorch-geometric.readthedocs.io/en/latest/

# Dataset Description

train.pt: contains molecule name/ID/graph index, node features, labels (some molecular property)
test.pt: contains molecule name/ID/graph index, node features only, labels (all set to 0.0)
sample_submission.csv: sample submission csv file

# Submissions required:

1) Submit your final code (submitted to Gradescope HW4: Programming)
2) Submit your predictions to Kaggle

# Evaluation

The evaluation metric for this competition is the mean absolute error regression loss (sklearn.metrics.mean_absolute_error).

# Important Note

Kaggle randomly separates the test set into two parts: public test and private test, in this HW the test set was separated evenly with no intersection. The split is fixed but not visible. Once you have submitted your prediction, Kaggle will show the accuracy on the public part of the test set immediately, while the accuracy on the private part of the test set will be hidden until the end of this homework. This means your final accuracy and rank (private test) is not necessarily the same as you will see during the competition (public test).
