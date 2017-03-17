from __future__ import print_function

from mlp_train import load_samples
from nutsflow import Take, Consume, Concat
from nutsml import PrintTypeInfo, ViewImageAnnotation

train_samples, val_samples = load_samples()

(train_samples >> Concat(val_samples) >> Take(10) >> PrintTypeInfo() >>
 ViewImageAnnotation(0, 1, pause=1) >> Consume())
