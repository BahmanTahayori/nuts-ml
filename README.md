# nuts-ml

Flow-based data preprocessing for (GPU/deep) machine learning.

API documentation and tutorials can be found here:  
https://maet3608.github.io/nuts-ml/


# Installation

## Virtual environment

```
$ pip install virtualenv
$ cd my_projects
$ virtualenv vnuts
```

### Activate/Deactivate virtual environment

Linux, Mac:  
```
$ source vnuts/bin/activate
$ deactivate
```

Windows:  
```
> vnuts\Scripts\activate.bat
> vnuts\Scripts\deactivate.bat
```


## Nuts-ml

1) Activate virtual environment (if not already active)
```
cd my_projects
source vnuts/bin/activate
```

2) Clone git repo
```
cd vnuts
git clone https://github.com/maet3608/nuts-ml
```

3) Install package with dependencies and run unit tests

`pip install -r requirements.txt`

or manually

```
pip install dplython
pip install pyyaml
pip install numpy
pip install matplotlib
pip install xlrd
pip install pandas
pip install pillow
pip install scikit-image
pip install scipy
```

If you encounter the following error when installing `scikit-image` under
windows:
`
distutils.errors.DistutilsError: Setup script exited with error: 
Microsoft Visual C++ 9.0 is required. Get it from http://aka.ms/vcpython27
`  

Then download Windows binaries for scikit-image from: 
`http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-image`
and install via
```pip install scikit_image-0.12.3-cp27-cp27m-win_amd64.whl```

If you encounter the following error when installing `scipy` under
windows:
`
numpy.distutils.system_info.NotFoundError: no lapack/blas resources found
`
Then download Windows binaries for scikit-image and Numpy+MKL from:
`http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy`
`http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy`
and install via
```pip install numpy‑1.11.2+mkl‑cp27‑cp27m‑win_amd64.whl```
```pip install scipy-0.18.1-cp27-cp27m-win_amd64.whl```



```
cd nuts-ml
python setup.py install
pytest
```

4) Run Python interpreter and import ```nutsml``` 
```
python
>>> import nutsml
>>> nutsml.__version__
'1.0.1'
>>> exit()
```

5) Try tiny example
```
python
>>> from nutsflow import *
>>> from nutsml import *
>>> samples = [('pos', 1), ('pos', 1), ('neg', 0)]
>>> stratify = Stratify(1, mode='up')
>>> samples >> stratify >> Collect()
[('neg', 0), ('pos', 1), ('neg', 0), ('pos', 1)]
>>> exit()
```

# Contribution guidelines

## Requirements

For every function
- unit tests
- documentation

## Naming conventions

### Nuts

Nuts are implemented as classes but operate as functions. Due to this
hybrid status names of nuts should use CamelCase (like classes) but 
also should describe an action (like functions).

Good examples:
- TransformImages
- LogCols
- ImageMean

Bad examples:
- ImageTransformer, transform_images
- Logger
- calc_image_mean


### Modules and packages

Use singular for module and package names, e.g. reader.py 
