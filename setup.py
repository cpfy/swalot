import os
from os.path import dirname, join as pjoin
from setuptools import setup, find_packages

meta = {}
with open(pjoin('swalot', '__version__.py')) as f:
    exec(f.read(), meta)

setup(
    name=meta['__title__'],
    version=meta['__version__'],
    description=meta['__description__'],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url=meta['__url__'],
    author=meta['__author__'],
    author_email=meta['__contact__'],
    license=meta['__license__'],
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'GPUtil',
        'rich',
    ],
)