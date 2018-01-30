# Preprocess-CIFAR
A tool for converting CIFAR-10 and CIFAR-100 datasest into PNG images with additional preprocessing options such as grayscaling.

## Introduction
The tools provided are compatible with [CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) datasets which contains 32x32 images that are a subset of the [80 Million Tiny Images](http://people.csail.mit.edu/torralba/tinyimages/) dataset. 

CIFAR-10 dataset contains 60,000 32x32 colour images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. CIFAR-100 is similar to CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class.

Benchmarks for the CIFAR-10 and CIFAR-100 datasets, and others can be found [here]https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html).

### Preprocessing
The training and test datasets are provided in files that contain a 'pickled' object produced with [cPickle](http://www.python.org/doc/2.5/lib/module-cPickle.html) by the authors.
They are then loaded into Numpy arrays, and the features are separated from the labels.

The data is then converted to images into `training` and `testing` directories. The format
for the filename is as follows: `TYPE_RANDOM_LABEL_LABELCOUNT.png`

- `TYPE`: Indicates dataset type, could be either `train` or `test`
- `RANDOM`: Short randomly generated UUID-style characters e.g. `7daa28`
- `LABEL`: The groundtruth label for the image (between 0-9)
- `LABELCOUNT`: The count for how many times a label was seen to easily

This format is useful for quickly extracting information about the dataset and target labels from the filename, while ensuring that each image's filename is unique.

## Getting Started

### Requirements
- Python 2.7+

### Installation

Install the Python dependencies using pip: `pip install -r REQUIREMENTS.txt`

### Usage

#### CIFAR-10
The training data in CIFAR-10 comes in 5 different 'batch' files, while the testing data comes in a single file. Before starting, ensure that you have the `data_batch_1`, `data_batch_2`, `data_batch_3`, `data_batch_4`, `data_batch_5` and `test_batch` provided [here](https://www.cs.toronto.edu/~kriz/cifar.html). The script accepts a folder path as input the directory containing the necessary files. The script also assumes the output directory exists so ensure that you have a designated output directory for the preprocessed images as it will not be created automatically.

To preprocess the training set, use the following:

`python src/cifar10.py --dataset train --input_folder /path/to/pickled/files --output_path /path/to/output/training`

To preprocess the test set, use the following:

`python src/cifar10.py --dataset train --input_folder /path/to/pickled/files --output_path /path/to/output/testing`

**Note:** We assume that the filenames are kept intact from the original dataset. If they have been renamed, the constants can be easily changed inside `src/cifar10.py` to the appropriate filename.

#### CIFAR-100
Unlike CIFAR-10, the CIFAR-100 dataset comes in a single file for the training set and a sinlge file for the test set. Before starting, ensure that you have the `train` and `test` provided [here](https://www.cs.toronto.edu/~kriz/cifar.html). The script accepts a folder path as input the directory containing the necessary files. The script also assumes the output directory exists so ensure that you have a designated output directory for the preprocessed images as it will not be created automatically.

To preprocess the training set, use the following:

`python src/cifar100.py --dataset train --input_file /path/to/train --output_path /path/to/output/training`

To preprocess the test set, use the following:

`python src/cifar100.py --dataset train --input_file /path/to/test --output_path /path/to/output/testing`

#### Grayscale

The original images are coloured, you may optionally pass the `--grayscale` parameter to convert the images to grayscale.

#### Logging

You may optionally pass the `--logging info` parameter to display the progress of the script, which looks like this:

```
...
[utils.py:120 - preprocess() - INFO] Step #6000: saved train_b9c487_42_65.png
[utils.py:120 - preprocess() - INFO] Step #7000: saved train_8961fc_14_71.png
[utils.py:120 - preprocess() - INFO] Step #8000: saved train_a300ef_91_76.png
[utils.py:120 - preprocess() - INFO] Step #9000: saved train_23da3c_42_93.png
[utils.py:120 - preprocess() - INFO] Step #10000: saved train_686758_48_103.png
[utils.py:120 - preprocess() - INFO] Step #11000: saved train_56e075_64_130.png
[utils.py:120 - preprocess() - INFO] Step #12000: saved train_fa3998_53_121.png
[utils.py:120 - preprocess() - INFO] Step #13000: saved train_378111_17_105.png
...
```

## Reference
[Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Alex Krizhevsky, 2009.
