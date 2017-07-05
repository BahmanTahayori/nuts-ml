import os

from nutsflow.base import NutFunction
from config import Config
from os.path import join, exists, isdir, getmtime

"""
.. module:: checkpoint
   :synopsis: Creation of checkpoints with network weights and parameters.
"""

class Checkpoint(NutFunction):
    def __init__(self, create_network, basedir='checkpoints', **config):
        self.basedir = basedir
        self.create_network = create_network
        self.config = Config(**config)
        self.config.bestscore = None

    def datapaths(self, checkpointname):
        configpath = join(self.basedir, checkpointname, 'config.json')
        weightspath = join(self.basedir, checkpointname, 'weights')
        return configpath, weightspath

    def dirs(self):
        """
        Return full paths to all checkpoint folders.

        :return: Paths to all folders under the basedir.
        :rtype: list
        """
        dirs = (join(self.basedir, d) for d in os.listdir(self.basedir))
        return [d for d in dirs if isdir(d)]

    def latest(self):
        """
        Find most recently modified/created checkpoint folder.

        :return: Full path to checkpoint folder if it exists otherwise None.
        :rtype: str | None
        """
        dirs = sorted(self.dirs(), key = getmtime, reverse=True)
        return dirs[0] if dirs else None

    def checkpoint_exits(self, checkpointname):
        return exists(join(join(self.basedir, checkpointname)))

    def load(self, checkpointname='checkpoint'):
        configpath, weightspath = self.datapaths(checkpointname)
        checkpoint_exits = self.checkpoint_exits(checkpointname)
        if checkpoint_exits:
            self.config.load(configpath)
        self.network = self.create_network(self.config)
        if checkpoint_exits:
            self.network.load_weights(weightspath)
        return self.network, self.config

    def save(self, checkpointname='checkpoint'):
        configpath, weightspath = self.datapaths(checkpointname)
        self.network.save_weights(weightspath)
        self.config.save(configpath)
        return join(self.basedir, checkpointname)

    def save_best(self, score, checkpointname='checkpoint', isloss=False):
        bestscore = self.config.bestscore
        if (bestscore is None
            or (isloss and score < bestscore)
            or (not isloss and score > bestscore)):
            self.config.bestscore = bestscore
            self.save(checkpointname)
        return join(self.basedir, checkpointname)

    def __call__(self, score):
        self.savebest(score)
        return score

# def create_network(config):
#     eta = config.eta
#     pass
#
#
# checkpoint = Checkpoint(create_network, eta=0.01)
#
# network, config = checkpoint.load()  # most recent or provide specific path
#
# for epoch in EPCOHS:
#     trainerr = train_network
#     valerr = validate_network
#
#     checkpointfolder = checkpoint.save()
#     checkpointfolder = checkpoint.savebest(valerr)
#
#     for image in misclassified_images:
#         save_image(checkpointfolder, image)
#
#     network.evaluate() >> checkpoint.saveBest() >> Collect()
