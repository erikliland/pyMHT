init:
	sudo apt-get install python3-setuptools
	sudo easy_install3 pip
	sudo -H pip install -r requirements.txt


test:
	nosetests tests
