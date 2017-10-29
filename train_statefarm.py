#!/usr/bin/env python

import argparse
import os

from dataprovider import DataProvider, TrainingDataProvider
from precompute import FeatureProvider
from model import ModelBuilder

def clip(values, minval, maxval):
    total = 0.0
    for i in range(len(values)):
        if values[i] < maxval:
            values[i] += minval
        else:
            values[i] = maxval
        total = sum(values)
    return [v / total for v in values]

def chunk(values, n):
    for i in xrange(0, len(values), n):
        yield values[i:i+n]

def limit_range(val, low, high):
    if val > high:
        return high
    elif val < low:
        return low
    return val

def predict_states(model, data_provider, batch_size, filename):
    test_batches = data_provider.get_batches('test', batch_size=batch_size)
    filename_batches = chunk(test_batches.filenames, batch_size)

    outclasses = ['c{0}'.format(i) for i in range(10)]
    with open(filename, 'w') as fout:
        fout.write('img,' + ','.join(outclasses) + '\n')
        for b in range(int(79726 / batch_size)):
            imgs, _ = next(test_batches)
            filenames = next(filename_batches)
            label_lookup = {}
            probabilities, labels = model.predict_all(imgs)
            for i, l in enumerate(labels):
                label_lookup[l] = i
            for p, i in zip(probabilities, filenames):
                values = clip([p[label_lookup[l]] for l in outclasses],
                              minval=0.05 / (len(label_lookup) - 1),
                              maxval=0.95)
                line = '{0},'.format(i[8:]) + ','.join(['{0:.4f}'.format(v) for v in values]) + '\n'
                fout.write(line)
                print(line.strip())

def main(path, is_training, is_predicting, model_weights_file, submission_file):
    print('Starting train_statefarm.py')
    print('* using path: {0}'.format(path))
    print('* training: {0}, predicting: {1}'.format(is_training, is_predicting))

    batch_size = 64
    data_provider = DataProvider(os.path.join(path, 'partition'), batch_size)
    feature_provider = FeatureProvider(data_provider)
    training_data_provider = TrainingDataProvider(data_provider, feature_provider)

    builder = ModelBuilder(
        training_data_provider,
        dropout=0.6,
        batch_size=batch_size)

    if is_training:
        print('Train last layer of dense model with batch normalization.')
        builder.train_last_layer()

    if is_training:
        print('Train dense layers of model with batch normalization.')
        builder.train_dense_layers()

    model = builder.build(data_provider)

    if not is_training:
        print('Loading model weights from {0}'.format(model_weights_file))
        model.load_weights(data_provider.get_weight_filepath(model_weights_file))
    else:
        model.train()
        print('Writing model weights to {0}'.format(model_weights_file))
        model.save_weights(data_provider.get_weight_filepath(model_weights_file))

    if is_predicting:
        print('Writing predictions to {0}'.format(submission_file))
        batch_size = 2
        data_provider = DataProvider(path, batch_size)
        predict_states(model, data_provider, batch_size, submission_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        dest='train',
        action='store_true',
        help='If specified, trains the model.')
    parser.add_argument(
        '--predict',
        dest='predict',
        action='store_true',
        help='If specified, predicts classifications.')
    parser.add_argument(
        '--path',
        dest='path',
        required=True,
        help='The directory location containing training/validation/test data.')
    parser.add_argument(
        '--weightfile',
        dest='weightfile',
        required=True,
        help='File to store/read model weights from.')
    parser.add_argument(
        '--submission',
        dest='submission',
        required=True,
        help='Filename of the submission file to write predictions to.')
    args = parser.parse_args()
    main(args.path, args.train, args.predict, args.weightfile, args.submission)
