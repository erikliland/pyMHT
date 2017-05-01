#OS := $(shell uname)

ifeq ($(shell uname),Darwin)
#Run macOS commands
init:
	echo "macOS"
	sudo -H pip3 install --upgrade pip
	sudo -H pip3 install -r requirements.txt
	sudo python3 setup.py install

ifeq ($(shell uname),Linux)
#Run Linux commands
init:
	echo "Linux"
	sudo apt-get update
	sudo apt-get upgrade
	sudo apt-get install python3-dev
	sudo apt-get install python3-setuptools
	sudo easy_install3 pip
	sudo apt-get install python3-tk
	sudo apt-get install python-glpk
	sudo apt-get install glpk-utils
	sudo apt-get install python-numpy
	sudo -H pip install -r requirements.txt
	sudo python3 setup.py install
