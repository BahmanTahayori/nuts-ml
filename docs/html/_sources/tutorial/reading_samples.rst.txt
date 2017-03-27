.. _reader:

Reading samples
===============

**nuts-ml** does not have a specific class or data structure defining data samples
buts many nuts nuts return ``tuples`` and expect indexable data structures such as
``tuples`` or ``lists`` as input.

Here a toy example to demonstrate some basic operation before moving on to more advanced topics. Given the file ``tests/data/and.csv`` that contains the truth table of the logical ``and``
operation

.. code::

  x1,x2,y
  0,0,no
  0,1,no
  1,0,no
  1,1,yes

we can load its lines via Python's ``open`` function and collect them in a ``list``

  >>> from nutsflow import *

  >>> open('tests/data/and.csv') >> Collect()
  ['x1,x2,y\n', '0,0,no\n', '0,1,no\n', '1,0,no\n', '1,1,yes']

However, what is typically needed is tuples with the data and no header. 
We therefore drop the first line, split all remaining lines at ',' and remove the 
pesky newline character (``\n``) 

  >>> Split = nut_function(lambda s : s.strip().split(','))
  >>> load_data = open('tests/data/and.csv')
  >>> load_data >> Drop(1) >> Split() >> Print() >> Consume()
  ['0', '0', 'no']
  ['0', '1', 'no']
  ['1', '0', 'no']
  ['1', '1', 'yes']

Better but all numbers in the first and second column are strings. We use
``MapCol`` and ``int`` to convert them to integers and also make the loading 
of the data more generic

  >>> Load = nut_source(lambda fname: open('tests/data/'+fname))
  >>> (Load('and.csv') >> Drop(1) >> Split() >> MapCol((0,1), int) >>
  ... Print() >> Consume())
  (0, 0, 'no')
  (0, 1, 'no')
  (1, 0, 'no')
  (1, 1, 'yes')


