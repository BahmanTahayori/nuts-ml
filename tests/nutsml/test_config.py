"""
.. module:: test_config
   :synopsis: Unit tests for config module
"""

import pytest

import nutsml.config as nc

expected_repr = """{
  "a": "1",
  "b": {
    "c": 2
  }
}"""


def test_Config():
    cfg = nc.Config({'name': 'stefan', 'address': {'number': 12}})
    assert cfg['name'] == 'stefan'
    assert cfg.name == 'stefan'
    assert cfg['address']['number'] == 12
    assert cfg.address.number == 12

    cfg.address.number = 7
    assert cfg['address']['number'] == 7
    assert cfg.address.number == 7


def test_repr():
    cfg = nc.Config({'a': '1', 'b': {'c': 2}})
    print(cfg.__repr__())
    assert cfg.__repr__() == expected_repr


def test_isjson():
    assert nc.Config.isjson('mydir/somefile.json')
    assert nc.Config.isjson('mydir/somefile.JSON')
    assert not nc.Config.isjson('mydir/somefile.yaml')


def test_save_load():
    cfg = nc.Config({'number': 13, 'name': 'Stefan'})

    cfg.save('tests/data/configuration.yaml')
    newcfg = nc.Config()
    loaded_cfg = newcfg.load('tests/data/configuration.yaml')
    assert newcfg.number == 13
    assert newcfg == cfg
    assert loaded_cfg == cfg

    cfg.save('tests/data/configuration.JSON')
    newcfg = nc.Config()
    newcfg.load('tests/data/configuration.JSON')
    assert newcfg.number == 13
    assert newcfg == cfg


def test_load_config():
    cfg = nc.load_config('tests/data/config.yaml')
    assert cfg.filepath == 'c:/Maet'
    assert cfg['imagesize'] == [100, 200]

    with pytest.raises(IOError) as ex:
        nc.load_config('does not exist')
    assert str(ex.value).startswith('Configuration file not found')
