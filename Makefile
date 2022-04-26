# Define environment, bash script execution line

python-default: script
	#
	# Python dependencies (pip)
	# First two commands to help with installation on RIT CS systems
	pip install --user numpy --upgrade
	pip install --user packaging --upgrade
	pip install --user python-terrier bs4 tqdm pandas lxml --upgrade
	pip install --upgrade git+https://github.com/terrierteam/pyterrier_colbert

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

data: collection post-data raw-post-data math-data
	
collection:
	wget https://www.cs.rit.edu/~dprl/data/ARQMath/ARQMath_Collection.zip
	unzip ARQMath_Collection.zip
	rm ARQMath_Collection.zip

# Tokenized with Terrier defaults (stopwords + porter stemmer), English tokenizer
post-data:
	wget https://www.cs.rit.edu/~dprl/data/ARQMath/ARQMath_Collection-post-ptindex.zip
	unzip ARQMath_Collection-post-ptindex.zip
	rm ARQMath_Collection-post-ptindex.zip

# No stopwords or stemming, using English tokenizer
raw-post-data:
	wget https://www.cs.rit.edu/~dprl/data/ARQMath/ARQMath_Collection-post-ptindex-rawtokens.zip
	unzip ARQMath_Collection-post-ptindex-rawtokens.zip
	rm ARQMath_Collection-post-ptindex-rawtokens.zip


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

baseline:
	./run-topics-2020
	./run-topics-2021

experiment:
	python3 src/play_with_data.py ./ARQMath_Collection-math-ptindex ./ARQMath_Evaluation/topics_task_1/2020_topics_task1.xml ./ARQMath_Evaluation/qrels_task_1/2020_qrels_task1.tsv