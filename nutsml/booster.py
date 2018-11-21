"""
.. module:: booster
   :synopsis: Boosting of wrongly predicted samples
"""

import numpy as np

from random import random
from nutsflow import nut_processor, Tee, Collect, Flatten, Print


@nut_processor
def Boost(iterable, batcher, network):
    """
    iterable >> Boost(batcher, network)

    Boost samples with high softmax probability for incorrect class.
    Expects one-hot encoded targets and softmax predictions for output.

    | network = Network()
    | build_batch = BuildBatch(BATCHSIZE).input(...).output()
    | boost = Boost(build_batch, network)
    | samples >> boost >> build_batch >> network.train() >> Consume()

    :param iterable iterable: Iterable with samples.
    :param nutsml.BuildBatch batcher: Batcher used for network training.
    :param nutsml.Network network: Network used for prediction
    :return: Generator over samples to boost
    :rtype: generator
    """

    def do_boost(probs, target):
        assert len(target) > 1, 'Expect one-hot encoded target: ' + str(target)
        assert len(target) == len(probs), 'Expect softmax probs: ' + str(probs)
        return random() > probs[np.argmax(target)]

    #print("len(iterable)", len(iterable))
    samples1, samples2 = iterable >> Tee(2)
    for batch in samples1 >> batcher:
        inputs, targets = batch
        print("inputs[0].shape", inputs[0].shape)
        print("targets[0].shape", targets[0].shape)
        preds = inputs >> network.predict() >> Print() >> Collect()
        print("len(preds)", len(preds))
        for p, t, s in zip(preds, targets[0], samples2):
            print("p, t, s", p, t, s)
            if do_boost(p, t):
                yield s
