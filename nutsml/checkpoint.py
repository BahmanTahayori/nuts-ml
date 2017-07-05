import os

from os.path import join, exists, isdir, getmtime
from nutsflow.base import NutFunction
from . config import Config


"""
.. module:: checkpoint
   :synopsis: Creation of checkpoints with network weights and parameters.
"""


class Checkpoint(NutFunction):
    """
    A factory for checkpoints to periodically save network weights and other
    hyper/configuration parameters.

    | Example usage
    | def create_network(config):
    |   model = Sequential()
    |   ...
    |   optimizer = opt.SGD(lr=config.lr)
    |   model.compile(optimizer=optimizer, metrics=['accuracy'])
    |   return KerasNetwork(model)
    |
    |
    | checkpoint = Checkpoint(create_network, lr=0.001)
    | network, config = checkpoint.load()
    |
    | for epoch in xrange(EPOCHS):
    |   train_err = train_network()
    |   val_err = validate_network()
    |
    |   if epoch > 10:
    |     config.lr = config.lr / 2
    |
    |   checkpoint.save_best(val_err)
    |

    Checkpoints can also be saved under different names, e.g.

    |  checkpoint.save_best(val_err, 'checkpoint'+str(epoch))

    And specific checkpoints can be loaded:

    | network, config = checkpoint.load('checkpoint103')

    If no checkpoint is specified the most recent one is loaded.
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

    def save(self, checkpointname='checkpoint'):
        """
        Save network weights and configuration under the given name.

        :param str checkpointname: Name of checkpoint folder. Path will be
           self.basepath/checkpointname
        :return: path to checkpoint folder
        :rtype: str
        """
        configpath, weightspath = self.datapaths(checkpointname)
        self.network.save_weights(weightspath)
        self.config.save(configpath)
        return join(self.basepath, checkpointname)

    def save_best(self, score, checkpointname='checkpoint', isloss=False):
        """
        Save best network weights and config under the given name.

        :param float|int score: Some score indicating quality of network.
        :param str checkpointname: Name of checkpoint folder.
        :param bool isloss: True, score is a loss and lower is better otherwise
           higher is better.
        :return: path to checkpoint folder
        :rtype: str
        """
        bestscore = self.config.bestscore
        if (bestscore is None
            or (isloss and score < bestscore)
            or (not isloss and score > bestscore)):
            self.config.bestscore = bestscore
            self.save(checkpointname)
        return join(self.basepath, checkpointname)

    def load(self, checkpointname=None):
        """
        Create network, load weights and configuration.

        :param str|none checkpointname: Name of checkpoint to load. If None
           the most recent checkpoint is used. If no checkpoint exists yet
           the network will be created but no weights loaded and the
           default configuration will be returned.
        :return: (network, config)
        :rtype: tuple
        """
        configpath, weightspath = self.datapaths(checkpointname)
        if configpath:
            self.config.load(configpath)
        self.network = self.create_network(self.config)
        if weightspath:
            self.network.load_weights(weightspath)
        return self.network, self.config

    def __call__(self, accuracy):
        """
        Enables checkpoint to be used in nuts-ml flow.

        samples >> build_batch >> network.evaluate() >> checkpoint >> Consume()

        :param float|int accuracy: Some measure of network accuracy
           (NOT loss!)
        :return: accuracy
        :rtype: int|float
        """
        self.savebest(accuracy)
        return accuracy

