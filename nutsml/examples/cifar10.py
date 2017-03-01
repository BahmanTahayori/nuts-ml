"""
Example nuts-ml pipeline for CIFAR-10 training and prediction
"""

import urllib
import os
import tarfile
import cPickle

import numpy as np

from keras.metrics import categorical_accuracy
from datetime import datetime
from nutsflow import *
from nutsml import *

NUM_EPOCHS = 100
BATCH_SIZE = 32
NUM_CLASSES = 10
INPUT_SHAPE = (3, 32, 32)


def download(destfolder='temp'):
    def progress(blocknum, bs, size):
        print '\rdownloading %.0f%%' % (100.0 * blocknum * bs / size),

    tarfilename = 'cifar-10-python.tar.gz'
    tarfilepath = os.path.join(destfolder, tarfilename)
    untar_folder = os.path.join(destfolder, 'cifar-10')
    url = 'https://www.cs.toronto.edu/~kriz/' + tarfilename

    if not os.path.exists(tarfilepath) and not os.path.exists(untar_folder):
        urllib.urlretrieve(url, tarfilepath, progress)

    print 'untarring...',
    if not os.path.exists(untar_folder):
        with tarfile.open(tarfilepath, 'r:gz') as tfile:
            tfile.extractall(path=untar_folder)
    print 'done.'

    if os.path.exists(tarfilepath):
        os.remove(tarfilepath)

    return untar_folder


def load_batch(batchfilepath):
    with open(batchfilepath, 'rb') as f:
        batch = cPickle.load(f)
    data, labels = batch['data'], batch['labels']
    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_names(untar_folder):
    filepath = os.path.join(untar_folder, 'cifar-10-batches-py', 'batches.meta')
    with open(filepath, 'rb') as f:
        return cPickle.load(f)['label_names']


def read_samples(untar_folder):
    batchpath = os.path.join(untar_folder, 'cifar-10-batches-py')
    from glob import glob
    for filepath in glob(batchpath + '/*_batch*'):
        data, labels = load_batch(filepath)
        fold = 'train' if 'data_batch' in filepath else 'test'
        for image, label in zip(data, labels):
            image = np.moveaxis(image, 0, 2)
            yield image, label, fold


def create_network():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return KerasNetwork(model)


def view(batchfolder):
    (read_samples(batchfolder) >> Take(1000) >> PrintTypeInfo()
     >> ViewImage(0, pause=0.5, figsize=(1, 1)) >> Consume())


def train(batchfolder):
    print 'creating network...'
    network = create_network()
    rerange = TransformImage(0).by('rerange', 0, 255, 0, 1, 'float32')
    build_batch = (BuildBatch(BATCH_SIZE)
                   .by(0, 'image', 'float32')
                   .by(1, 'one_hot', 'uint8', NUM_CLASSES))
    p = 0.5
    augment = (AugmentImage(0)
               .by('identical', 1.0)
               .by('brightness', p, [0.7, 1.3])
               .by('color', p, [0.7, 1.3])
               .by('shear', p, [0, 0.1])
               .by('fliplr', p)
               .by('rotate', p, [-10, 10]))
    plot_eval = PlotLines((0,1), layout=(2,1))

    label2name = load_names(batchfolder)
    train_samples, val_samples = (read_samples(batchfolder) >> rerange >>
                                  PartitionByCol(2, ['train', 'test']))

    print 'training...', len(train_samples), len(val_samples)
    start_time = datetime.now()
    for epoch in xrange(NUM_EPOCHS):
        print 'EPOCH:', epoch

        t_results = (train_samples >> PrintProgress(train_samples) >>
                     Pick(1) >> NOP(augment) >>
                     build_batch >> network.train() >> Collect())
        t_loss, t_acc = zip(*t_results)
        print "training loss    :\t\t{:.6f}".format(np.mean(t_loss))
        print "training acc     :\t\t{:.1f}".format(100 * np.mean(t_acc))

        metric = [categorical_accuracy]
        e_acc = (val_samples >> build_batch >> network.evaluate(metric))
        print "evaluation acc   :\t\t{:.1f}".format(100 * e_acc)

        plot_eval((np.mean(t_acc), e_acc))
    duration = datetime.now() - start_time
    print "training time    :\t\t{:.1f} min".format(duration.seconds / 60.0)

    # print 'predicting...',
    # show_image = ViewImageAnnotation(0, 1, pause=1)
    # pred_batch = (BuildBatch(BATCH_SIZE).by(0, 'image', 'float32'))
    # samples = val_samples >> Take(100) >> Collect()
    # predictions = (samples >> pred_batch >> network.predict() >>
    #                Map(np.argmax) >> Map(lambda l: label2name[l]))
    # samples >> Get(0) >> Zip(predictions) >> show_image >> Consume()


if __name__ == "__main__":
    batchfolder = download()
    train(batchfolder)
