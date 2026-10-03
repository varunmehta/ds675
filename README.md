# All exercises from [DS-675](https://web.njit.edu/~usman/courses/ds675/) by Prof. Usman

This is my scratchpad for notes from the class. As a I learn more, I keep adding.

From the course website - [DS-675](https://web.njit.edu/~usman/courses/ds675/)
> **Course Overview**
> This course is a hands-on introduction to machine learning and contains both theory and application. We will cover linear and non-linear classification methods, image classification, neural network training, logistic regression, feature selection, clustering, dimensionality reduction, and decision trees and random forests. We will also implement methods in Python with scikit-learn and Keras and if time permits study kernel methods, Bayesian learning, and autoencoders.

## Quick Start
 * Install [Python](https://www.python.org/downloads/) 
 * Install [VSCode](https://code.visualstudio.com/download) 
 * Install [Jupyter Notebook Plugin](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
 * Clone this repo
 * Run Notebooks

> For those who know, create a [`.venv`](https://docs.python.org/3/tutorial/venv.html) to manage dependencies. 
> 
> [Using `venv` with VSCode](https://code.visualstudio.com/docs/python/environments)

If you are very new to python and package management with too many dependencies to install one-by-one is overwhelming, download [Anaconda](https://www.anaconda.com/download). It bundles Python with hundreds of pre-installed data science tools. The only downside is that you're downloading software that you might not need.

## Notebooks

> More will be added as the class proceeds!

* [Linear Classifiers](/linear-classifiers/)
  * [Breast Cancer Data](/linear-classifiers/breast-cancer-training.ipynb)
  * [Optional Exercise 1](/linear-classifiers/optional_exercise_1.ipynb)
  * [Optional Exercise 2](/linear-classifiers/optional_exercise_2.ipynb)
* [Image Classification](/image-classification/) 
* [Neural Networks](/neural-networks/nn.ipynb)

## Project
This is my hands on project to learn comparison between SVM, Linear Models and Neural Networks. Do not plagiarize if you are attending this course!! 
  * [Project](/project/project-ds675.ipynb)

## Jupyter Notebooks
> If you hit upon this repo in lieu to learn, this is just a scratch pad, check other notebooks out there. 

Jupyter notebooks and Python notebooks are an important tool for data science. Jupyter Notebook, JupyterLab, Google Colab, and Kaggle Notebooks all use the exact same underlying architecture (mixing executable code cells with Markdown text cells), but they serve very different purposes depending on whether you want to work locally or in the cloud.

The primary differences boil down to where the code runs (your computer vs. cloud servers) and how you access resources like GPUs, datasets, and collaborative tools.

### Direct Comparison Overview 

| Feature | Jupyter Notebook | JupyterLab | Google Colab | Kaggle Notebooks  |
| --- | --- | --- | --- | --- |
| Environment | Local (Offline) | Local (Offline) | Cloud-hosted | Cloud-hosted  |
| Interface | Single notebook per tab | Tabbed, multi-file IDE | Single notebook per tab | Single notebook per tab  |
| Computation | Your hardware | Your hardware | Google Cloud | Kaggle/Google Servers  |
| Free GPU Access | No (Uses your hardware) | No (Uses your hardware) | Yes (Limited/Shared) | Yes (Generous weekly quotas)  |
| Setup Required | Yes (/Anaconda) | Yes (/Anaconda) | None (Google Account) | None (Kaggle Account)  |
| Data Storage | Local Hard Drive | Local Hard Drive | Google Drive | Kaggle Datasets & Scratchpad  |
| Collaboration | Hard (Git/Manual share) | Hard (Git/Manual share) | Easy (Google Docs style) | Easy (Shared notebooks/teams)  |

### Using Jupyter Locally

* [Jupyter Notebooks in VS Code](https://www.youtube.com/watch?v=suAkMeWJ1yE)
* [Jupyter Notebooks in PyCharm](https://www.youtube.com/watch?v=uiIKaacMGoE)
 
If you have not used Jupyter notebooks in the past, this is a good introduction video. 

* [Jupyter Notebook in 10 minutes](https://youtu.be/H9Iu49E6Mxs)
* [Jupyter Notebook Complete Beginner Guide - From Jupyter to Jupyterlab, Google Colab and Kaggle!](https://youtu.be/5pf0_bpNbkw)

###  Recommendation
* Choose JupyterLab if you are working on a commercial project with private data and already have a strong computer.
* Choose Google Colab if you want to jump right into coding immediately without installing software, or if you need to collaborate directly with teammates.
* Choose Kaggle Notebooks if you want to practice on public datasets or participate in AI and machine learning competitions.