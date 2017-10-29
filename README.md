# State Farm Distracted Driver Classifier

A classifier for the [State Farm Distracted Driver competition](https://www.kaggle.com/c/state-farm-distracted-driver-detection) written in Python using Keras.

## Running

    python train_statefarm.py --train --predict --path ../data --submission submission.csv --weightsfile model_weights.h5

The above assumes the following directory structure:

* data/statefarm/partition/train/{c0, ..., c9}/*.jpg
* data/statefarm/partition/valid/{c0, ..., c9}/*.jpg
* data/statefarm/test/unknown/*.jpg

In the above case, `--path` should point to the 'statefarm' directory.

The files `utils.py`, `vgg16.py` and `vgg16bn.py` are taken from the fast.ai github repository [here](https://github.com/fastai/courses/tree/master/deeplearning1/nbs).