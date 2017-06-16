# pyMHT 

## Track oriented, multi target, multi hypothesis tracker
Multi frame multi target tracking module with 2/2&m/n initialization algorithm and an AIS aided track oriented multi hypothesis tracking algorithm.


## Installation

You can get the latest and greatest from
[github](https://github.com/erikliland/pymht):

    $ git clone git@github.com:erikliland/pymht.git pymht
    $ cd pymht
    $ sudo python setup.py install


`pyMHT` depends on the following modules,

* `Cython`    	(for compiling Munkres algorithm)
* `numpy`     	(for core functionality)
* `scipy`     	(for core functionality)
* `matplotlib`	(for ploting)
* `pytest`		(for testing)
* `matplotlib`	(for ploting)
* `Munkres` [[Github](https://github.com/jfrelinger/cython-munkres-wrapper)]
*  `OR-TOOLS` (for solving ILP´s) [[Github](https://github.com/google/or-tools)]

All modules except `OR-TOOLS` can be installed via pip:

	$ pip install -r preRequirements.txt
	$ pip install -r requirements.txt
	
`OR-TOOLS` must be installed manually.

## Test instalation
To test the instalation run in the pyMHT directory:

		$ pytest
This module does not contain any scenarios or examples. This is placed in another repository [pyMHT-simulator](https://github.com/erikliland/pyMHT-simulator). 

## Background
This Python module is the result of a project assignment and a Master´s thesis

[Project report](https://mfr.osf.io/render?url=https://osf.io/2eeqd/?action=download%26mode=render)

[Thesis]()

## Build status
Master [![Build Status](https://travis-ci.org/erikliland/pyMHT.svg?branch=master)](https://travis-ci.org/erikliland/pyMHT)

Development [![Build Status](https://travis-ci.org/erikliland/pyMHT.svg?branch=development)](https://travis-ci.org/erikliland/pyMHT)

Master [![Coverage Status](https://coveralls.io/repos/github/erikliland/pyMHT/badge.svg?branch=master)](https://coveralls.io/github/erikliland/pyMHT?branch=master)