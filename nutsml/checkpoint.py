import os

from nutsflow.base import NutFunction
from config import Config
from os.path import join, exists, isdir, getmtime

"""
.. module:: checkpoint
   :synopsis: Creation of checkpoints with network weights and parameters.
"""


class Checkpoint(NutFunction):
    """
    A factory for checkpoints to periodically save network weights and other
    hyper/configuration parameters.

# def create_network(config):
#     eta = config.eta
#     return network
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

    """

    def __init__(self, create_network, basepath='checkpoints', **config):
        """
        Create checkpoint factory.

        >>> def create_network(config):
        ...     return 'network eta=' + str(config.eta)

        >>> checkpoint = Checkpoint(create_network, eta=0.1)
        >>> network, config = checkpoint.load()
        >>> network
        'network eta=0.1'
        >>> config.eta
        0.1

        :param function create_network: Function that takes a Config
           and returns a nuts-ml Network.
        :param string basedir: Path to folder that will contain
          checkpoint folders.
        :param kwargs config: Keyword arguments used to create a Config
          dictionary.
        """
        if not exists(basepath):
            os.makedirs(basepath)
        self.basepath = basepath
        self.create_network = create_network
        self.config = Config(**config)
        self.config.bestscore = None

    def dirs(self):
        """
        Return full paths to all checkpoint folders.

        :return: Paths to all folders under the basedir.
        :rtype: list
        """
        paths = (join(self.basepath, d) for d in os.listdir(self.basepath))
        return [p for p in paths if isdir(p)]

    def latest(self):
        """
        Find most recently modified/created checkpoint folder.

        :return: Full path to checkpoint folder if it exists otherwise None.
        :rtype: str | None
        """
        dirs = sorted(self.dirs(), key=getmtime, reverse=True)
        return dirs[0] if dirs else None

    def datapaths(self, checkpointname=None):
        """
        Return paths to network weights and config files.

        If no checkpoints exist under basedir (None, None) is returned.

        :param str|None checkpointname: Name of checkpoint. If name is None
           the most recent checkpoint is used.
        :return: (weightspath, configpath) or (None, None)
        :rtype: tuple
        """
        name = checkpointname
        path = self.latest() if name is None else join(self.basepath, name)
        if path is None or not exists(path):
            return None, None
        return join(path, 'config.json'), join(path, 'weights')

    def load(self, checkpointname=None):
        configpath, weightspath = self.datapaths(checkpointname)
        if configpath:
            self.config.load(configpath)
        self.network = self.create_network(self.config)
        if weightspath:
            self.network.load_weights(weightspath)
        return self.network, self.config

    def save(self, checkpointname='checkpoint'):
        configpath, weightspath = self.datapaths(checkpointname)
        self.network.save_weights(weightspath)
        self.config.save(configpath)
        return join(self.basepath, checkpointname)

    def save_best(self, score, checkpointname='checkpoint', isloss=False):
        bestscore = self.config.bestscore
        if (bestscore is None
            or (isloss and score < bestscore)
            or (not isloss and score > bestscore)):
            self.config.bestscore = bestscore
            self.save(checkpointname)
        return join(self.basepath, checkpointname)

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
