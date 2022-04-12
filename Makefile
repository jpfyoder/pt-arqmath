# Define environment, bash script execution line

python-default: script
	#
	# Python dependencies (pip)
	# First two commands to help with installation on RIT CS systems
	pip install --user numpy --upgrade
	pip install --user packaging --upgrade
	pip install --user python-terrier bs4 tqdm pandas lxml --upgrade

script:
	# Creating test script...
	@echo "#!`which bash`" > exec_line
	@cat exec_line bin/arqmath-test-TEMPLATE > arqmath-test
	@rm exec_line
	# Test script is ./arqmath-test

data:
	wget https://www.cs.rit.edu/~dprl/data/ARQMath/ARQMath_Collection.zip
	unzip ARQMath_Collection.zip
	rm ARQMath_Collection.zip

posts:
	./arqmath-test test/indexTest.xml -s

math:
	./arqmath-test test/indexTest.xml -m

clean:
	rm -rf *-ptindex 
