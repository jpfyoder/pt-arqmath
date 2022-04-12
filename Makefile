install:
	# First two lines to help with installation on RIT CS systems
	pip install --user numpy --upgrade
	pip install --user packaging --upgrade
	pip install --user python-terrier bs4 tqdm pandas

data:
	wget https://www.cs.rit.edu/~dprl/data/ARQMath/ARQMath_Collection.zip
	unzip ARQMath_Collection.zip
	rm ARQMath_Collection.zip

posts:
	export PYTHONPATH=./src:$PYTHONPATH
	./arqmath-test test/indexTest.xml -s

math:
	./arqmath-test test/indexTest.xml -m

clean:
	rm -rf *-ptindex
