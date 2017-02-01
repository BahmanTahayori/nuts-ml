import os
import sys
import glob
import shutil
import nutsml

from setuptools import setup, find_packages, Command
from setuptools.command.test import test as TestCommand


class CleanCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for folder in ['build', 'dist']:
            if os.path.exists(folder):
                shutil.rmtree(folder)
        for egg_file in glob.glob('*egg-info'):
            shutil.rmtree(egg_file)


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


setup(
    name='nutsml',
    version=nutsml.__version__,
    url='https://github.ibm.com/aur/nuts-ml',
    license='Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
    author='Stefan Maetschke',
    author_email='stefanrm@au1.ibm.com',
    description='Flow-based data preprocessing for Machine Learning',
    install_requires=[
        # TODO: declare nutsflow as dependency from git repo
        # 'nutsflow >= 1.0.0',
        'dplython >= 0.0.7',
        'numpy >= 1.11.0',
        'matplotlib >= 1.5.1',
        'scikit-image >= 0.12.3',
        'pandas >= 0.18.1',
        'pytest >= 3.0.3',
    ],
    tests_require=['pytest >= 3.0.3'],
    dependency_links=[
        "git@github.ibm.com:aur/nuts-flow.git",
        'git+ssh://github.ibm.com/aur/nuts-flow/blob/master/dist/nutsflow-1.0.0.tar.gz',
    ],
    platforms='any',
    packages=find_packages(exclude=['setup']),
    include_package_data=True,
    cmdclass={
        'test': PyTest,
        'clean': CleanCommand,
    },
    classifiers=[
        'Programming Language :: Python',
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
