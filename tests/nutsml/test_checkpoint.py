"""
.. module:: test_checkpoint
   :synopsis: Unit tests for checkpoint module
"""
import pytest
import os
import nutsml.checkpoint as nc

from os.path import join


@pytest.fixture(scope="function")
def checkpointdirs():
    basedir = 'tests/data/checkpoints'
    checkpoint1 = join(basedir, 'checkpoint1')
    checkpoint2 = join(basedir, 'checkpoint2')
    os.mkdir(checkpoint1)
    os.mkdir(checkpoint2)

    def fin():
        os.rmdir(checkpoint1)
        os.mkdir(checkpoint2)

    return checkpoint1, checkpoint2


def create_network(config):
    return 'network eta=' + str(config.eta)


def test_Checkpoint():
    checkpoint = nc.Checkpoint(create_network, eta=0.1)
    network, config = checkpoint.load()
    assert network == 'network eta=0.1'

def test_dirs_empty():
    basedir = 'tests/data/checkpoints'
    checkpoint = nc.Checkpoint(create_network, basedir=basedir)
    assert checkpoint.dirs() is []

def test_dirs(checkpointdirs):
    checkpoint1, checkpoint2 = checkpointdirs
    basedir = 'tests/data/checkpoints'
    checkpoint = nc.Checkpoint(create_network, basedir=basedir)
    assert sorted(checkpoint.dirs()) is [checkpoint1, checkpoint2]

# def test_latest():
#     basedir = 'tests/data/checkpoints'
#     checkpoint = nc.Checkpoint(create_network, basedir=basedir)
#
#     latest = checkpoint.latests()
#     assert latest is None
#
#     os.mkdir(join(basedir, 'checkpoint1'))
#     os.mkdir(join(basedir, 'checkpoint2'))
#
#     latest = checkpoint.find_latests()
#     assert latest == join(basedir, 'checkpoint2')
#
#     os.rmdir(join(basedir, 'checkpoint1'))
#     os.mkdir(join(basedir, 'checkpoint2'))
