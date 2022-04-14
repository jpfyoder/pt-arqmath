# Define environment, bash script execution line

python-default: script
	#
	# Python dependencies (pip)
	# First two commands to help with installation on RIT CS systems
	pip install --user numpy --upgrade
	pip install --user packaging --upgrade
	pip install --user python-terrier bs4 tqdm pandas lxml --upgrade

script:
	# Creating test scripts...
	@echo "#!`which bash`" > exec_line
	@cat exec_line bin/arqmath-test-TEMPLATE > arqmath-test
	@cat exec_line bin/run-topics-test-TEMPLATE > run-topics-test
	@cat exec_line bin/run-topics-2020-TEMPLATE > run-topics-2020
	@chmod u+x arqmath-test
	@chmod u+x run-topics-test run-topics-2020
	@rm exec_line
	# Quick run/eval test is ./run-topics-test
	# Evaluation run for ARQMath-1 topics w. BM25 is ./run-topics-2020
	# Indexing test script is ./arqmath-test

data:
	wget https://www.cs.rit.edu/~dprl/data/ARQMath/ARQMath_Collection.zip
	wget https://www.cs.rit.edu/~dprl/data/ARQMath/ARQMath_Collection-post-ptindex.zip
	unzip ARQMath_Collection.zip
	unzip ARQMath_Collection-post-ptindex.zip
	rm *.zip

posts:
	./arqmath-test test/indexTest.xml -s

math:
	./arqmath-test test/indexTest.xml -m

clean:
	rm -rf *-ptindex 
