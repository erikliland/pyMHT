from setuptools import setup, find_packages

setup(
    name = 'pyMHT',
    version = '0.9',
    author = 'Erik Liland',
    author_email = 'erik.liland@gmail.com',
    description = ('An implementation of a track oriented multi hypothesis tracker' +
      'with integer linear programming in Python'),
    license = 'BSD',
    keywords = 'mht tomht radar tracking track-split track split multi target multitarget',
    url = 'http://autosea.github.io/sf/2016/04/15/radar_ais/',
    packages = find_packages(exclude=['examples', 'docs']),
    include_package_data = True,
    install_requires = [
        'matplotlib',
        'numpy',
        'scipy',
    ],
)
