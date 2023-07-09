# PoseModel

<p align="center">
  <img src="https://github.com/MeysamAmirsardari/PoseModel/blob/main/UI/poseModel.jpg" width="270" height="270">
</p>

PoseModel: open-source toolkit for robust, markerless pose estimation and animal behavioral classification, Specialized for datasets with visual occlusion and special conditions.

### Under Construction...
* **Warning: This project is still under development and it's not yet ready for real-world usage!** Please wait for the first official release... :)
* more detail will be added after the first official release

This repository contains the implementation of the PoseModel project. This project proposes a novel method for automated classification of behavioral states in non-human models. Our approach leverages spatiotemporal graph modeling, combining variational graph autoencoder (VGAE), HDBSCAN clustering, and RNN seq2seq modeling.

<p align="center">
  <img src="https://github.com/MeysamAmirsardari/PoseModel/blob/main/UI/pma.jpg" style="max-width: 70%;">
</p>

<p align="center">
  <img src="https://github.com/MeysamAmirsardari/PoseModel/blob/main/UI/sam.png" style="max-width: 270;">
</p>


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



