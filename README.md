# cs6140_a3

## Collaborators

<b>Jake Van Meter</b>
- NUID: 002965845

<b>Yihan Xu</b>
- NUID: 001566238

## Structure of Project
- This project is consists of 10 python files, with main.py as the execution file.
- The pre-processing execution logic is in preprocessing.py, with import of the functions from utils.py, pca.py and regression.py.
- The training and evaluation logic is wrapped in a Classifier class in classifier.py.
- All datasets are in data folder.
- All images are in images folder.

## Instructions for Running
Instructions for running from CLI: python `<path/to/assignment2>/main.py <desired_task>`

Where you replace `<desired_task>` with one or more of the following options:
- `all` to run all tasks.
- `1` to run the preprocessing.
- `2` to run all of the classification.
- `knn` to run K-Nearest Neighbor classification.
- `lr` to run Linear Regression classification.
- `svm` to run Support Vector Machine classification (with iterations).
- `mnb` to run Multinomial Naive Bayes classification.
- `bnb` to run Bernoulli Naive Bayes classification.
- `gnb` to run Gaussian Naive Bayes classification (with iterations).
- `dt` to run Decision tree classification.

If no value is provided for `<desired_task>` all code (i.e. preprocessing and classification using all models) is automatically run.
Examples:

- Running all tasks: `python main.py all` or `python main.py`
- Running just preprocessing: `python main.py 1`
- Running all classifiers: `python main.py 2`
- Running just Support Vector Machine classification: `python main.py svm`
- Running just Decision Tree classification: `python main.py dt`
- Running a mixture of Logistic Regression and Gaussian Naive Bayes classification: `python main.py lr gnb`

The following dependencies were used and can be installed with `pip install <dependency>`:
- NumPy
- Pandas
- Sklearn (installed with pip install scikit-learn)
- Matplotlib
- Seaborn

See `requirements.txt` for a list of all dependencies including those installed with the above packages.

## Instructions on Extensions
The Extensions we have covered:
- We used 5 different classifiers on the dataset when.
- We have experimented with PCA, normalization, and mathematical transformation of the data set.
- We learned how to use lasso regularization for feature selection via identification of insignificant features.
- We learned how to use a random forest classifier to rank the importance of features.
- We carried out 3 iterations on our 2 top performing classifiers.
- We have generated ROC and calculated AUC for each classifier.

Please follow the instructions for running, all extensions will be covered.

## Travel days used
No travel day is used.

## Links

- <a href="https://scikit-learn.org/stable/modules/classes.html#">Scikit-Learn API Reference</a>

## Operating System and IDE

<b>Yihan Xu</b>
- <b>OS</b>: MacOS
- <b>IDE</b>: PyCharm

<b>Jake Van Meter</b>
- <b>OS</b>: Windows 10
- <b>IDE</b>: VSCode
