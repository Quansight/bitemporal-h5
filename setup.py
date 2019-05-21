#!/usr/bin/env python3
import os
import sys

from setuptools import setup


def main():
    """The main entry point."""
    if sys.version_info[:2] < (3, 4):
        sys.exit('xonsh currently requires Python 3.4+')
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r') as f:
        readme = f.read()
    skw = dict(
        name='bth5',
        description='Bitemporal HDF5',
        long_description=readme,
        description_content_type="text/markdown",
        license='BSD',
        version='0.0.0',
        author='Anthony Scopatz',
        maintainer='Quansight',
        author_email='scopatz@gmail.com',
        url='https://github.com/quansight/bitemporal-h5',
        platforms='Cross Platform',
        classifiers=['Programming Language :: Python :: 3'],
        packages=['bth5'],
        package_dir={'bth5': 'bth5'},
        zip_safe=True,
        )
    setup(**skw)


if __name__ == '__main__':
    main()
