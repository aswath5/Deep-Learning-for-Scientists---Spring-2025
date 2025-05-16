# Description

In this Homework, you will train a model (on protein sequences) to perform a multiclass classification task using the PFam seed dataset.

Part 1: Use the given pretrained model, i.e., Protein BERT, and fine-tune it ((https://huggingface.co/Rostlab/prot_bert)). This should be your baseline model.

Part 2: Try other models from Hugging Face and beat your baseline. Your Kaggle submission should come from your best-performing model.

Bonus Q: Use protein embeddings and visualize protein clusters with UMAP/tSNE.

You will be provided with the dataset and a skeleton code(s) that you must complete to conduct baseline training successfully, followed by fine-tuning.

Read more about the data here:

[1] Bileschi, M. L., Belanger, D., Bryant, D. H., Sanderson, T., Carter, B., Sculley, D., … & Colwell, L. J. (2022). Using deep learning to annotate the protein universe. Nature Biotechnology, 40(6), 932-937.

# Dataset Description

### Each csv contains the following fields:
sequence: Usually the input features to your model. Amino acid sequence for this domain. There are 20 very common amino acids, and 4 amino acids that are quite uncommon: X, U, B, O, Z.
family_accession: Accession number in form PFxxxxx.y (Pfam), where xxxxx is the family accession, and y is the version number. Some values of y are greater > 10, and so 'y' has two digits.
family_id: One word name for family. Use as labels for your model.
sequence_name: Sequence name, in the form "$uniprot_accession_id/$start_index-$end_index".
aligned_sequence: Contains a single sequence from the multiple sequence alignment.

# Submissions required:

1) A pdf report to summarize the background, methods, results (training/testing plots, etc.) and discussion (submitted to Gradescope HW3: Report)
2) Submit your final code (submitted to Gradescope HW3: Programming)
3) Submit your predictions to Kaggle

# Evaluation

The evaluation metric for this competition is accuracy score (sklearn.metrics.accuracy_score).

# Important Note

Kaggle randomly separates the test set into two parts: public test and private test, in this HW the test set was separated evenly with no intersection. The split is fixed but not visible. Once you have submitted your prediction, Kaggle will show the accuracy on the public part of the test set immediately, while the accuracy on the private part of the test set will be hidden until the end of this homework. This means your final accuracy and rank (private test) is not necessarily the same as you will see during the competition (public test).
