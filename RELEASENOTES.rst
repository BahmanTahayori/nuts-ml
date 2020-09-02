Release notes
=============

1.0.44
------
- safe loading of yaml config files
- PrintColType now prints PyTorch tensors

1.0.43
------
- axis_off and labels_off added to ViewImage
- every_sec, every_n added to ViewImage
- long_description added to setup.py

1.0.42
------
- batchstr() added
- build_tensor_batch now supports expand_dims

1.0.41
------
- edge filter added as transformer and augmentation
- travis config for Python 3.7 fixed
- Wrapper for Pytorch models added


1.0.40
------
- support for Python 3.4 dropped
- support for Python 3.7 added
- catching edge case SplitRandom((1,0))


1.0.39
------
- bug in network.predict fixed for multi-input networks
- boosting fixed
- Config is now OrderedDict

1.0.38
------
- debug output for batcher added
- fmt parameter for batcher removed
- mixup augmentation added: `batcher.Mixup()`
- deprecated `as_grey` by skimage has been replaced by `as_gray`


1.0.37
------
- Start of release notes