"""
Setup script for pyMHT by Erik Liland 2017
"""
from setuptools import find_packages
import os

if ('USE_CYTHON' in os.environ) and (int(os.environ['USE_CYTHON']) == 1):
    print("Using Cython")
    USE_CYTHON = True
else:
    print("NOT using Cython")
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'

if USE_CYTHON:
    from distutils.extension import Extension
    import numpy
    extensions = [Extension("pymht.utils.cKalman", ["pymht/utils/cKalman" + ext]),
                  Extension("pymht.utils.cFunctions", ["pymht/utils/cFunctions" + ext],
                            include_dirs=[numpy.get_include()])]
else:
    extensions = None

name = "pyMHT"
version = "1.0"
author = "Erik Liland"
author_email = "erik.liland@gmail.com"
description = "A track oriented multi hypothesis tracker with integer linear programming"
license = "BSD"
keywords = 'mht tomht radar tracking track-split track split multi target multitarget'
url = 'http://autosea.github.io/sf/2016/04/15/radar_ais/'
install_requires = ['matplotlib', 'numpy', 'scipy', 'psutil', 'termcolor']

packages = find_packages(exclude=['examples', 'docs'])
print("Packages", packages)

if USE_CYTHON:
    from distutils.core import setup
    from Cython.Build import cythonize
    print("Cythonize extensions")
    extensions = cythonize(extensions)
    print("Setup with external modules")
    setup(
        name=name,
        version=version,
        author=author,
        author_email=author_email,
        description=description,
        license=license,
        keywords=keywords,
        url=url,
        packages=packages,
        # include_package_data = True,
        # install_requires = install_requires,
        ext_modules=extensions,
    )
else:
    from setuptools import setup
    setup(
        name=name,
        version=version,
        author=author,
        author_email=author_email,
        description=description,
        license=license,
        keywords=keywords,
        packages=packages,
        # include_package_data=True,
        # install_requires=install_requires,
    )
