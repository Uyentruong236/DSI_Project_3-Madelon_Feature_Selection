# Madelon Data: Feature Selection and Classification

## Project Description

This repository includes all the files related to my analysis of the Madelon dataset as well as a synthetic dataset of 220,000 data points with 1000 features. 

**Madelon data description from the UCI Website:** 

"Madelon is a two-class classification problem with continuous input variables. The difficulty is that the problem is multivariate and highly non-linear... Madelon is an artificial dataset containing data points grouped in 32 clusters placed on the vertices of a five dimensional hypercube and randomly labeled +1 or -1. The five dimensions constitute 5 informative features. 15 linear combinations of those features were added to form a set of 20 (redundant) informative features. Based on those 20 features one must separate the examples into the 2 classes (corresponding to the +-1 labels). We added a number of distractor feature called 'probes' having no predictive power. The order of the features and patterns were randomized."

### Project goals:

To develop a series of models for two purposes:

1) for the purposes of identifying relevant features.

2) for the purposes of generating predictions from the model.

### Content: 

In the ipynb folder, start with **Project_3_Madelon_Report.ipynb** - This report describes the process and results of the my data analysis. The report includes the following sections:

- EDA of each subset,
- Benchmarking results,
- Methods to identify Salient Features and results,
- Methods to identify Feature Importances and results,
- Results from Gridsearch and Model Pipelines.


