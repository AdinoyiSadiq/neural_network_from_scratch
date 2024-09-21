# Neural Network From Scratch

This repository implements a neural network from scratch with just Python without the use of frameworks such as Tensorflow or Pytorch. 
This follows the implementation found in NNFS [book](https://github.com/Sentdex/nnfs_book).

## Setup 

1. Clone the repo to your local machine
2. Navigate to the root of the repo and setup your virtual environment using `virtualenv nnfs --python=python`.This assumes that you have both `Python` and `virtualenv` installed.
3. Run `source nnfs/bin/activate` to activate the virtual environment.
4. Install the packages with `pip install -r requirements.txt`.
5. Run `python data-download.py` and then `python model/start.py` to download the datasets and run the model respectively. 