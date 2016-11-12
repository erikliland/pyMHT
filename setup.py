from setuptools import setup, find_packages

setup(
    name = 'tomht',
    version = '0.1',
    author = 'Erik Liland',
    author_email = 'erik.liland@gmail.com',
    description = ('An implementation of a track oriented multi hypothesis tracker' +
      'with integer linear programming in Python'),
    license = 'BSD',
    keywords = 'mht tomht radar tracking track-split track split multi target multitarget',
    url = 'http://autosea.github.io/sf/2016/04/15/radar_ais/',
    packages = find_packages(),
    package_data={'pykalman': ['datasets/descr/robot.rst', 'datasets/data/robot.mat']},
    classifiers = [
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: BSD License',
      'Operating System :: OS Independent',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    include_package_data = True,
    install_requires = [
      'numpy',
      'scipy',
      'matplotlib',
    ],
    tests_require = [
      'nose',
    ],
    extras_require = {
        'docs': [
          'Sphinx',
          'numpydoc',
        ],
        'tests': [
          'nose',
        ],
    },
)
