install:
	pip install python-terrier bs4 html tqdm pandas

data:
	wget https://www.cs.rit.edu/~dprl/data/ARQMath/ARQMath_Collection.zip
	unzip ARQMath_Collection.zip
	rm ARQMath_Collection.zip

posts:
	./arqmath-index post_data/rawTest.xml -s

math:
	./arqmath-index post_data/rawTest.xml -m

clean:
	rm -rf *-ptindex
