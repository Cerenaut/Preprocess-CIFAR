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

from __future__ import print_function

import os
import logging
import numpy as np

import utils

TRAIN_DATA_FILENAMES = [
    'data_batch_1',
    'data_batch_2',
    'data_batch_3',
    'data_batch_4',
    'data_batch_5'
]

TEST_DATA_FILENAME = 'test_batch'


def setup_arg_parsing():
    """
    Parse the commandline arguments
    """
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--dataset', dest='dataset', required=True,
                        help='The type of dataset could be (train|test|valid)')

    parser.add_argument('--input_folder', dest='input_path', required=True,
                        help='Path to the folder containing pickedl files')

    parser.add_argument('--output_path', dest='output_path', required=True,
                        help='Path to folder for saving generated images')

    parser.add_argument('--class_list', dest='class_list', required=False,
                        help='List of classes to be selected'
                             '(default=%(default)s).')


    parser.add_argument('--nb_class', dest='nb_class', required=False,
                        help='List of maximum number of elements per class to be selected, -1 for no limit'
                             '(default=%(default)s).')

    parser.add_argument('--grayscale', dest='grayscale', action='store_true',
                        required=False, help='Convert images to grayscale '
                                             '(default=%(default)s).')

    parser.add_argument('--logging', dest='logging', required=False,
                        help='Logging level (default=%(default)s). '
                             'Options: debug, info, warning, error, critical')

    parser.set_defaults(class_list='[0,1,2,3,4,5,6,7,8,9]')
    parser.set_defaults(nb_class='[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]')
    parser.set_defaults(grayscale=False)
    parser.set_defaults(logging='warning')

    return parser.parse_args()


def check_args(args):
    """
    Validates the arguments

    :param args: The commandline arguments
    """
    if not os.path.exists(args.input_path):
        logging.error('The input path is not valid: ' + args.input_path)
        exit(1)

    if not os.path.isdir(args.output_path):
        logging.error('The output path is not valid: ' + args.output_path)
        exit(1)


def parse_train_data(input_path):
    features = []
    labels = []

    # Load pickled source file
    try:
        for filename in TRAIN_DATA_FILENAMES:
            data = utils.unpickle(os.path.join(input_path, filename))
            features.append(data['data'])
            labels.append(data['labels'])
    except Exception as ex:
        logging.error('Failed to load input files from: ' + args.input_path)
        logging.error('Exception: %s', ex)
        exit(1)

    # Prepare features
    features = np.concatenate(features)
    features = features.reshape(features.shape[0], 3, 32, 32)
    features = features.transpose(0, 2, 3, 1).astype('uint8')

    # Prepare labels
    labels = np.concatenate(labels)

    return features, labels


def parse_test_data(input_path):
    # Load pickled source file
    try:
        data = utils.unpickle(os.path.join(input_path,
                                           TEST_DATA_FILENAME))
    except Exception as ex:
        logging.error('Failed to load input files from: ' + args.input_path)
        logging.error('Exception: %s', ex)
        exit(1)

    # Prepare features
    features = data['data']
    features = features.reshape(features.shape[0], 3, 32, 32)
    features = features.transpose(0, 2, 3, 1).astype('uint8')

    # Prepare labels
    labels = np.asarray(data['labels'])

    return features, labels


def main():
    """
    The main scope of the preprocessor containing the high level code
    """

    args = setup_arg_parsing()

    # Setup logging
    log_format = ("[%(filename)s:%(lineno)s - %(funcName)s() " +
                  "- %(levelname)s] %(message)s")
    logging.basicConfig(format=log_format,
                        level=utils.logger_level(args.logging))

    # Validate args
    check_args(args)

    labels = None
    features = None

    if args.dataset == 'train':
        features, labels = parse_train_data(args.input_path)
    elif args.dataset == 'test':
        features, labels = parse_test_data(args.input_path)
    else:
        logging.error('Only "train" and "test" dataset types are supported.')
        exit(1)

    # Start preprocessing images
    utils.preprocess(args.dataset, features, labels, args.output_path,
                     args.grayscale, args.class_list, args.nb_class)

if __name__ == '__main__':
    main()
