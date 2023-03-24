# MLCourse: Week6

We will look at a few examples of Unsupervised Learning methods in this week. So far, we have seen a bunch of Supervised machine learning(ML) methods such as DNN, CNN deploying for Regression or classification tasks. These algorithms need true label of the objects that are there in dataset for classification task.

There is another genre of machine learing methods which goes as Unsupervised ML. They learn from the dataset without any true labels. These algorithms are capable of finding structure or hidden patterns in any dataset.

In today's session we will learn about,

- <mark>KMeans clustering algorithm</mark>
  - Apply this to the standard IRIS Dataset (from sklearn library).

- <mark>DBSCAN clustering algorithm</mark>
  - Apply this algorithm and KMeans on a Particle Physics problem to find cluster of particles and benchmark their performances

- <mark>Dimension Reduction Algorithms(PCA and UMAP)</mark>
  - Do we need multiple features to solve a classification task?
  - How are they helpful?
  - Can we compress the multiple features to a few features(Latent space or low dimensional space)? and check how objects from same class distributes themselves in this low dimensional space?
  - We will apply PCA and UMAP algorithms to the WZ vs ZZ classification task from week4 and benchmarks this algorithms


# Notebooks and Input Files
 This folder has,

- input/week6.txt: Input file for the particle clustering problem
- KMeans_IrisDataset.ipynb
- HEP_ParticlesHitProblem_KMeans_DBSCAN_notebook.ipynb
- PCA_UMAP_DimReductionNotebook.ipynb
- LectureSlides
- README.md


Please follow the lecture slides for more details and homework!

# Tasks

### KMeans and DBSCAN

1. Try KMeans algorithm with all four features of the iris dataset

2. Apply DBSCAN on the iris dataset problem

3. Use non-euclidean metric and check how the clustering process get affected. Benchmark your results.

4. Include Energy of the particles in the DBSCAN algorithm and check if there is any change in the no of cluster

5. Try to play with the weight on each features used for calculating loss functions and benchmark the results.
   - This may require to go deeper in the loss functions, making custom loss functions that you can use during training.
   - If you try yourself and fail, I can help you with this. Feel free to drop an email and visit my office.

### PCA and UMAP

1. Reduce 12 dimension to 3 dimension and visualize WZ , ZZ events in 3D space

2. Reduce 12 dimension to 5 dimension and visualize WZ , ZZ events in 2D space of different latent variables

3. Conduct a training with 5 latent variables and compare the ROC

4. Check different distance metric for these clustering algorithms. Tabulate your results and draw conclusion from your findings.

```Advanced```

What happens if you use only 100 samples of WZ and 10K samples of ZZ file? Train a NN with this extreme asymmetric class scenerio. Check if the results are stable or not. Then apply UMAP/PCA to check if you can separate WZ from ZZ more efficienty than NN or not.


```An Interesting Exercise```

5. Do we need all the 12 variables to beging with? You can start with top 5 or 7 variables and compress them to 2 dimensions.Check if there any substantial improvement in the clustering. Tabulate your results with different set up and make sense of the outcome.
