"""
.. module:: test_checkpoint
   :synopsis: Unit tests for checkpoint module
"""

import nutsml.checkpoint as nc

def create_network(config):
    return 'network eta=' + str(config.eta)

def test_Checkpoint():
    checkpoint = nc.Checkpoint(create_network, eta=0.1)
    network, config = checkpoint.load()
    assert network == 'network eta=0.1'

def test_find_latest():
    checkpoint = nc.Checkpoint(create_network, foldername='tests/data')
    latest = checkpoint.find_latests()
    assert latest