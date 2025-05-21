# ML Course IISER Pune
- PH6232: Machine Learning and its physics applications

- Course Website: https://alaha999.github.io/

- Course Modules: https://alaha999.github.io/modules/

## Installing packages
We need ```python```,```numpy```,```matplotlib```,```tensorflow```,```keras```, and ```scikit-learn``` for this course.
- How to install scikit-learn: [install scikit-learn](https://scikit-learn.org/stable/install.html#installing-the-latest-release)
- How to install tensorflow: [install_tf_documentation](https://www.tensorflow.org/install), [pypi_pip](https://pypi.org/project/tensorflow/), [how-to-install-python-tensorflow-in-windows](https://www.geeksforgeeks.org/how-to-install-python-tensorflow-in-windows/)
- How to install Keras: [install keras](https://pypi.org/project/keras/)

NB: Please make sure you have these libraries **working** in your system. If not, make sure you come to Wednesday's tutorial session, and we will help you with the installation! But this comes with a free joke,

![](https://i.pinimg.com/474x/8c/81/cd/8c81cd6b6744c99f04c07c6fb2616304.jpg)

### Recommended Workflow
Use the conda distribution to streamline your workflow and get rid of frequent anxiety attacks due to package dependency errors, etc. The steps are the following,

1. **Install conda:** [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) [ You can use miniconda or mamba whatever you like]
2. **Make a conda environment:** [Managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) (say the name is ```ml_course```)
3. **Activate the conda environment:** ```conda activate ml_course```
4. Then inside this environment, install the packages: ```conda install <package-name>```

*Note: Each time you want to work, you activate the environment and proceed. Installing packages is only for once.*


## Summary of the topics covered
|Week  | Materials |
|------|-----------|
|Week 1| - Python basics, matplotlib, numpy, pandas dataframe, etc. <br> - Lectures on neural networks (overview and maths)<br>|
|Week 2| - Regression using DNN in the context of projectile motion: Lectures and hands-on session|
|Week 3| - Classification using DNN in the context of projectile motion: Lectures and hands-on session|
|Week 4| - Assignments: WZ vs ZZ classification (LHC problem) and Gravitation Wave related classification (LIGO)|
|Week 5| - Convolutional neural network (CNN): Lectures and hands-on session (Low Pt vs High Pt track)|
|Week 6| - Unsupervised ML and Dimension reduction techniques|
|Week 7| - Generative Adversarial Networks (GAN)|
|Week 8| - Guest Lectures: Anomaly detection in the finance sector and Quantum ML|

```Author:```
Contact Arnab Laha (arnab.laha@cern.ch) for any details. Or visit office A-365, Main Building Second Floor.