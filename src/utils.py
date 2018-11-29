# Copyright (C) 2018 Project AGI
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import pickle
import logging
import binascii
import numpy as np

from PIL import Image

NUM_RANDOM = 10
DISPLAY_STEP = 1000


def logger_level(level):
    """
    Map the specified level to the numerical value level for the logger

    :param level: Logging level from command argument
    """
    try:
        level = level.lower()
    except AttributeError:
        level = ""

    return {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }.get(level, logging.WARNING)


def unpickle(filepath):
    """
    Loads a file with a 'pickled' object and returns a dictionary

    :param level: The path to a pickled file
    """
    with open(filepath, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data


def init_label_count(num_labels):
    """
    Initialises the label count dictionary with 0 values
    """
    label_count = {}

    for i in range(num_labels):
        label_count[i] = 0

    return label_count


def generate_filename(dataset, label, label_count):
    """
    Generates a randomised filename for an image

    :param dataset:
    :param label: The groundtruth label of the image
    :param label_count: The number of times label has been seen already
    """
    random = binascii.hexlify(os.urandom(NUM_RANDOM // 2)).decode()
    # filename = '%s_%s_%i_%i.png' % (dataset, random, label, label_count)
    filename = '%s.png' % (random)
    return filename


def preprocess(dataset, features, labels, target_path, grayscale=False, class_list='[0,1,2,3,4,5,6,7,8,9]', nb_class=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]):
    """
    Map the specified level to the numerical value level for the logger

    :param dataset: The dataset type which could be train or test
    :param features: The features (X) of the dataset
    :param labels: The labels (y) of the dataset
    :param target_path: The path to saving the file
    """
    num_labels = len(np.unique(labels))
    label_count = init_label_count(num_labels)

    if dataset == 'train':
        train = True
    else:
        train = False

    # Get size of the data
    size = features.shape[0]

    class_list = class_list.replace('[','')
    class_list = class_list.replace(']','')
    class_list = [int(s) for s in class_list.split(',')]

    nb_class = nb_class.replace('[','')
    nb_class = nb_class.replace(']','')
    nb_class = [int(s) for s in nb_class.split(',')]

    for i in range(size):
        label = labels[i]
        if label in class_list:
            count = 0
            if label in label_count:
                count = label_count[label]
                count += 1
            label_count[label] = count
            if nb_class[label]==-1 or count<=nb_class[label]:
                filename = generate_filename(dataset, label, label_count[label])
                if train:
                    new_target_path = os.path.join(target_path,str(label))
                    filepath = os.path.join(new_target_path, filename)
                else:
                    filepath = os.path.join(target_path, filename)
                    if nb_class[label]==1:
                        print(filename)
                image = Image.fromarray(features[i])

                if grayscale:
                    image = image.convert('L')
                else:
                    image = image.convert('RGB')

                image.save(filepath)

                if i % DISPLAY_STEP == 0 or i == 1:
                    logging.info('Step #%i: saved %s', i, filename)
