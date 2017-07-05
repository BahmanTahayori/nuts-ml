"""
.. module:: test_checkpoint
   :synopsis: Unit tests for checkpoint module
"""
import pytest
import os
import time
import nutsml.checkpoint as nc

from os.path import join

BASEPATH = 'tests/data/checkpoints'


@pytest.fixture(scope="function")
def checkpointdirs(request):
    basedir = 'tests/data/checkpoints'
    checkpoint1 = join(basedir, 'checkpoint1')
    checkpoint2 = join(basedir, 'checkpoint2')
    os.mkdir(checkpoint1)
    time.sleep(0.1)  # ensure diff in creation time of checkpoints
    os.mkdir(checkpoint2)

    def fin():
        os.rmdir(checkpoint1)
        os.rmdir(checkpoint2)

    request.addfinalizer(fin)
    return checkpoint1, checkpoint2


def create_network(config):
    return 'network eta=' + str(config.eta)


def test_Checkpoint():
    basepath = 'tests/data/tmp'
    checkpoint = nc.Checkpoint(create_network, basepath, eta=0.1)
    network, config = checkpoint.load()
    os.rmdir(basepath)
    assert network == 'network eta=0.1'


def test_dirs_empty():
    checkpoint = nc.Checkpoint(create_network, BASEPATH)
    assert checkpoint.dirs() == []


def test_dirs(checkpointdirs):
    checkpoint1, checkpoint2 = checkpointdirs
    checkpoint = nc.Checkpoint(create_network, BASEPATH)
    assert sorted(checkpoint.dirs()) == [checkpoint1, checkpoint2]


def test_latest_empty():
    checkpoint = nc.Checkpoint(create_network, BASEPATH)
    assert checkpoint.latest() is None


def test_latest(checkpointdirs):
    checkpoint1, checkpoint2 = checkpointdirs
    checkpoint = nc.Checkpoint(create_network, BASEPATH)
    assert checkpoint.latest() == checkpoint2


def test_datapaths_empty():
    checkpoint = nc.Checkpoint(create_network, BASEPATH)
    assert checkpoint.datapaths('blah') == (None, None)
