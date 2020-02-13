import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

VERSION = open('debayer/__init__.py').readlines()[-1].split()[-1].strip('\'')

setup(
    name='pytorch-debayer',
    version=VERSION,
    description='Convolutional PyTorch debayering layer.',
    author='Christoph Heindl',
    url='https://github.com/cheind/pytorch-debayer/',
    license='MIT',
    install_requires=required,
    packages=['debayer', 'debayer.apps'],
    include_package_data=True,
    keywords='debayer bayer pytorch convolution'
)
