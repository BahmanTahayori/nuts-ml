Introduction
============

Pipeline
--------

Software for GPU-based machine learning, specifically on image data,
tends to have a certain, common structure visualized by the following 
*canonical pipeline*

.. image:: pics/pipeline.png

When large data sets cannot be loaded into memory, data is instead read in 
small batches or single records that are processed individually through a
sequence of common operations/components

- *Reader*: sample data stored in CSV files, `Pandas <http://pandas.pydata.org/>`_ 
  tables or other formats is read,

- *Splitter*: samples are split into training, validation and sets, and stratified
  if necessary,

- *Loader*: image data is loaded for each sample when needed,

- *Transformer*: images are transformed, e.g. cropped or resized,

- *Augmenter*: images are augmented to increase data size by random rotations,
   flipping, changes to contrast, or others,

- *Batcher*: the transformed and augmented images are organized in mini-batches 
   for GPU processing,

- *Network*: a neural network is trained, evaluated or tested,

- *Logger*: and finally the network performance (loss, accuracy, ...)
   is logged or plotted.

Depending on the actual task (training, testing, evaluation, ...), data type
(image, video, text), or application some of the processing steps may differ but 
in general many components can be shared. 


nuts-ml
-------

**nuts-ml** is a library for flow-based data processing that provides common
operations as independent units, so called 'nuts'. Nuts can be freely arranged 
to build data flows that are efficient, easy to understand and easy to change.

The following example gives a taste of a **nuts-ml** data flow:

.. code:: python

  train_samples >> PrintProgress(train_samples) >>
    load_image >> transform >> augment >> Shuffle(100) >>
    build_batch >> network.train() >> Consume()


Architecture
------------

**nuts-ml** is based on `nuts-flow <https://github.com/maet3608/nuts-flow>`_,
which is described `here <https://maet3608.github.io/nuts-flow/>`_ .

.. image:: pics/architecture.png
