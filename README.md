# Kaggle

## Utils
The subfolder *util* contains datastructures that are used to generically manipulate with data and allow to reduce boilerplate code in every new project.

Import as follows:

    import os
	import sys	
	current_dir = os.getcwd()
	util_path = os.path.join(os.path.dirname(current_dir), '', 'util')
	sys.path.append(util_path)
	import Dataset as ds
	import DatasetModifier as dsm
	import Classifications as classifications
	...


## Environment
Anaconda is recommended because it makes it easier to setup and handle different datasciency python environments

For Nvidia GPU setup:
Install cuda toolkit 10
https://developer.nvidia.com/cuda-10.0-download-archive

Install latest NVIDIA GPU drivers

Via conda, create a new virtual environment and install the following:

conda create -n tensorflow2 python=3.6 cudnn cudatoolkit=10.0

conda activate tensorflow2

Check the following link for the latest tensorflow 2.0 version: https://www.tensorflow.org/install/gpu#tensorflow_20_rc

pip install tensorflow-gpu==2.0.0-rc1

conda install -c anaconda numpy

conda install -c anaconda scikit-learn
