from __future__ import print_function
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io
import codecs
import os
import sys

import rehline_po

here = os.path.abspath(os.path.dirname(__file__))

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.md')

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
    name='rehline_po',
    version=rehline_po.__version__,
    author='Alibek Orazalin',
    author_email='alibekorazalin@cuhk.edu.hk',
    url='https://github.com/softmin/ReHLine-PO/',
    description='Mini-portfolio optimization package fueled by ReHLine',
    long_description=long_description,
    packages=['rehline_po'],
    tests_require=['pytest'],
    install_requires=["rehline", "requests", "numpy"],
    cmdclass={'test': PyTest},
    include_package_data=True,
    # test_suite='tests.test_sandman',
    extras_require={
        'testing': ['pytest'],
    }
)