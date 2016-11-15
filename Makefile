OS := $(shell uname)
CPLEX := $(shell command -v ~/Applications/IBM/ILOG/CPLEX_Studio1263/cplex/bin/x86-64_osx/cplex -f 2> /dev/null)

ifeq ($(OS),Darwin)
#Run macOS commands
init:	
	sudo -H pip3 install --upgrade pip
	sudo -H pip3 install -r requirements.txt
	brew install homebrew/science/glpk
	sudo python3 setup.py install
	brew install wget
	wget -r -np -R *html,index.* -nH --cut-dirs=2 http://folk.ntnu.no/eriklil/mac/solvers/
ifndef CPLEX
	brew cask install java
	sudo chmod +x solvers/cplex*
	sudo ./solvers/cplex*
endif
	
	#if [ ! -d $("Applications/Gurobi*")]; \
	#then $(shell sudo installer -pkg solvers/gurobi* -target /); \
	#fi;

else
#Run Linux commands
init:
	sudo apt-get install python3-setuptools
	sudo easy_install3 pip
	sudo apt-get install python3-tk
	sudo -H pip install -r requirements.txt
	sudo apt-get install wget
	wget -nc -r -np -R *html,index.* -nH --cut-dirs=2 http://folk.ntnu.no/eriklil/linux/solvers/
	chmod +x solvers/cplex*
	sudo ./solvers/cplex*
	

endif