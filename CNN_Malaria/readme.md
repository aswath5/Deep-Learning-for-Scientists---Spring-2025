# Description

Malaria, a disease caused by protozoan parasites of the genus Plasmodium, is not only an acute life threat in many developing countries but also a significant burden on the healthcare system worldwide. Prompt and effective diagnostic methods are essential for the management and control of malaria. However, traditional diagnosis techniques like staining thin and thick peripheral blood smears and rapid test methods like OptiMAL, ICT require labor and high-cost toolkits. Thus, the efficiency may be limited due to time consumption, cost-effectiveness, labor intensiveness, etc [1]. In recent years, researchers have been developing automated screening methods with the help of machine learning methods. It has been shown that with the help of computer vision techniques, we can detect malaria based on the images of blood cells [2]. Furthermore, CNN-based models utilized for malaria detection have demonstrated notable performance improvements and can be readily trained with the help of pre-existing models [3].

[1] Tangpukdee N, Duangdee C, Wilairatana P, et al. Malaria diagnosis: a brief review[J]. The Korean journal of parasitology, 2009, 47(2): 93.
[2] Das D K, Ghosh M, Pal M, et al. Machine learning approach for automated screening of malaria parasite using light microscopic images[J]. Micron, 2013, 45: 97-106.
[3] Rajaraman S, Antani S K, Poostchi M, et al. Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images[J]. PeerJ, 2018, 6: e4568.

# Summary:

You are required to implement a convolutional neural network-based machine learning model to predict whether the cell in the given picture is parasitized by the genus Plasmodium or not. Please note the difference in the dataset from previously mentioned article; all pictures in the given dataset contain only a single cell which makes the classification task easier. All pictures in the training set are labeled with either 0 (uninfected) or 1 (parasitized). You can build a model from scratch or use a pre-trained model with other modifications. Once the model is ready, use it to make predictions on the test set and submit your predictions on Kaggle.

# Bonus Qs:

Extract the learned embeddings (e.g., from convolutional filters or MLP layers) and visualize them. Do you notice any meaningful patterns in the learned filters?
Plot tSNE/UMAP based on these embeddings. Are healthy and sick cells clearly distinguishable?
Submissions required:

1) A pdf report to summarize the background, methods, results and discussion (submit to Gradescope HW2: Report)
2) Submit your final code (submit to Gradescope HW2: Programming)
3) Submit your predictions to Kaggle

# Evaluation

The evaluation metric for this competition is mean F1-score (sklearn.metrics.f1_score).

# Important Note

Kaggle randomly separates the test set into two parts: public test and private test, in this HW the test set was separated evenly with no intersection. The split is fixed but not visible. Once you have submitted your prediction, Kaggle will show the accuracy on the public part of the test set immediately, while the accuracy on the private part of the test set will be hidden until the end of this homework. This means your final accuracy and rank (private test) is not necessarily the same as you will see during the competition (public test).
