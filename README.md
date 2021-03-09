This repository contains the code for the paper [Multi-Layer Semantic and Geometric Modeling with Neural Message Passing in 3D Scene Graphs for Hierarchical Mechanical Search](https://ai.stanford.edu/mech-search/hms/).

## Installation
To install the code, just follow these steps:

```
git clone https://github.com/StanfordVL/HMS.git
cd HMS
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
pip install -e .
```
To install older versions of pytorch with a corresponding CUDA (for example CUDA 11):

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch


## Data
Unfortunately due to having proprietary CAD files, we cannot release the data used for training publicly. To request a private copy, please email andreyk@stanford.edu

