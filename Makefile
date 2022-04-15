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
	@cat exec_line bin/arqmath-index-TEMPLATE > arqmath-index
	@chmod u+x arqmath-index
	@cat exec_line bin/run-topics-test-TEMPLATE > run-topics-test
	@cat exec_line bin/run-topics-2020-TEMPLATE > run-topics-2020
	@cat exec_line bin/run-topics-2021-TEMPLATE > run-topics-2021
	@chmod u+x run-topics-test run-topics-2020 run-topics-2021
	@rm exec_line
	#
	# Quick run/eval test script is ./run-topics-test
	# Evaluation run script for ARQMath-1 topics w. BM25 is ./run-topics-2020
	# Evaluation run script for ARQMath-2 topics w. BM25 is ./run-topics-2021
	#
	# Indexing test script is ./arqmath-test

data: collection post-data math-data
	
collection:
	wget https://www.cs.rit.edu/~dprl/data/ARQMath/ARQMath_Collection.zip
	unzip ARQMath_Collection.zip
	rm ARQMath_Collection.zip

post-data:
	wget https://www.cs.rit.edu/~dprl/data/ARQMath/ARQMath_Collection-post-ptindex.zip
	unzip ARQMath_Collection-post-ptindex.zip
 	rm ARQMath_Collection-post-ptindex.zip


math-data:
	wget https://www.cs.rit.edu/~dprl/data/ARQMath/ARQMath_Collection-math-ptindex.zip
	unzip ARQMath_Collection-math-ptindex.zip
	rm ARQMath_Collection-math-ptindex.zip

posts:
	./arqmath-index test/indexTest.xml -s

math:
	./arqmath-index test/indexTest.xml -m

eval:
	./run-topics-test

delete-results:
	rm -g *.res.gz

delete-indices:
	rm -rf *-ptindex 
