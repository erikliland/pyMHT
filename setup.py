"""
Setup script for pyMHT by Erik Liland 2017
"""
from setuptools import find_packages
import os
import sys
from setuptools import setup

if sys.version_info.major < 3:
    sys.exit('Sorry, Python 2 is not supported')

name = "pyMHT"
version = "2.0"
author = "Erik Liland"
author_email = "erik.liland@gmail.com"
description = "A track oriented multi hypothesis tracker with integer linear programming"
license = "BSD"
keywords = 'mht tomht radar tracking track-split track split multi target multitarget'
url = 'http://autosea.github.io/sf/2016/04/15/radar_ais/'
install_requires = ['matplotlib', 'numpy', 'scipy', 'psutil', 'termcolor', 'Cython']
packages = find_packages(exclude=['examples', 'docs'])

setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    description=description,
    license=license,
    keywords=keywords,
    packages=packages,
    install_requires=install_requires
)
