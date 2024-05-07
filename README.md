# PoseModel

<p align="center">
  <img src="https://github.com/MeysamAmirsardari/PoseModel/blob/main/UI/poseModel.jpg" width="270" height="270">
</p>

PosModel: Robust Behavioral Latent Embedding Toolkit Based on
Spatio‑Temporal Graph Modeling

### Under Construction...
* **Warning: This project is still under development and it's not yet ready for real-world usage!** Please wait for the first official release... :)
* more detail will be added after the first official release

This repository contains the implementation of the PoseModel project. This project aims to uncover complex behavioral patterns in a novel way by utilizing robust Graph Neural Networks (GNNs) for latent embedding. Through spatiotemporal graph modeling, the proposed method captures intricate relationships in behavioral data, enabling more accurate and insightful analysis.

<p align="center">
  <img src="https://github.com/MeysamAmirsardari/PoseModel/blob/main/UI/pma.jpg" style="max-width: 70%;">
</p>

<p align="center">
  <img src="https://github.com/MeysamAmirsardari/PoseModel/blob/main/UI/sam.png" style="max-width: 270;">
</p>

## What is this?
Quantifying and interpreting behavior is crucial to reveal various aspects of biological systems involved in behavioral mechanisms. An accurate and measurable description of behavior is essential for many research scenarios in neuroscience, neuroethology, and behavioral science.

We present a model for unsupervised latent embedding, clustering, and dynamics analysis of behavioral data. Our innovative spatio-temporal modeling approach utilizes pose-estimated visual data, transforming the variational embedding problem into a graph representation learning task. Leveraging the power of graph neural networks, we introduce the PoseModel, a graph representation learning tool based on a novel architecture as a variational-attentional-graph autoencoder. 

The tool's efficiency and agility are enhanced by the ability of the model to be independently applied to pose-estimated recordings of behavior. By considering the natural dependencies and stochastic nature of the data, our modeling method ensures robustness and flexibility across diverse test conditions and experimental settings.

We prepared a dataset of 460 frames featuring the behavior of two caged monkeys recorded under different conditions to train our pose-estimation model. The model achieved 9.47 pixels RMS error in HD recording, accurately estimating the poses of the monkeys in 25 hours of recorded behavior. This 25-hour dataset is used to train and evaluate the effectiveness of the PoseModel in the next stage.

The embedding model showed a satisfactory performance on data from monkeys, mice, and humans, and enables behavioral clustering and dynamics analysis. The embeddings are consistent across sessions and different animals and prove valuable in downstream tasks like behavioral state estimation and hierarchical analysis.

## Setup
### Install prerequisite libraries

To run PoseModel, install it locally using this:

```
wget https://github.com/MeysamAmirsardari/PoseModel/blob/main/requirements
```

Pip install libraries:
```
pip install -r requirements.txt
```

### Download and unzip contents from GitHub repo
Download and unzip contents from this repo and open up the command prompt and traverse to the location where you unzipped the repo contents

### Launch PoseModel!
Use this command to run it on your local machine
```
streamlit run PoseModel.py
```



