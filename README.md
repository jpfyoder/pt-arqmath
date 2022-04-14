## PyTerrier Framework for ARQMath Task 1

`pt-arqmath` was created for the Information Retrieval course at the Rochester Institute of Technology in Spring 2022. PyTerrier provides a flexible framework for building, running, and comparing a variety of different search engines, including neural retrieval models. 

This code is provided to make getting started with ARQMath Task 1 (Answer Retrieval) a bit easier. The rest of this document covers:

* Installation of the git package 
* Getting started
* Important notes (especially for CSCI 539 students)
* Search and evaluation using the full ARQMath collection

## Installation

**Please check the notes below** before following the installation instructions.

This code has been tested on MacOS X and Linux, and requires a bash command shell (i.e., command line). 

1. **Download** the code from GitLab using:  
`git clone https://gitlab.com/dprl/pt-arqmath.git`
2. If possible, make sure to install lxml, e.g., on Ubuntu, using: `sudo apt-get install lxml`
2. Enter the project directory using: `cd pt-arqmath`
3. Issue `make` to install PyTerrier and Python dependencies
4. Issue `make data` to download the collection of ARQMath posts

**Notes for RIT CS Students (Spring 2022)**

* It will probably be easiest to run the code on a Ubuntu system or virtual machine/environment (e.g., an RIT CS lab machine).
* If you receive a message complaining that a package is incompatible/too old, you can use:  
```
pip install --user <pkgname> --upgrade
```  
to update, and
```
pip install --user <pkgname>==X.Y.Z 
```
to select a specific package version (where X.Y.Z is a specific version number, e.g., 0.8.2).


## Getting Started

Some quick indexing and retrieval tests are provided by the `arqmath-test` bash script. The script has flags you can modify, for example to return index statistics, the lexicon produced after tokenization, whether to produce an index for posts/formulas/both, and a flag to control tokenization by PyTerrier  (e.g., stemming and stopword removal).

If you issue `./arqmath-test` without arguments, you should see the following:

```
usage: index_arqmath.py [-h] [-m | -mp] [-l] [-s] [-t TOKENS] [-d] xmlFile

Indexing tool for ARQMath data.

positional arguments:
  xmlFile               ARQMath XML file to index

optional arguments:
  -h, --help            show this help message and exit
  -m, --math            create only the math index
  -mp, --mathpost       create math and post indices
  -l, --lexicon         show lexicon
  -s, --stats           show collection statistics
  -t TOKENS, --tokens TOKENS
                        set tokenization property (none: no stemming/stopword removal)
  -d, --debug           include debugging outputs
```


First, to test indexing and retrieval for complete posts, issue:

```
make posts
```
On the terminal, you will see information about the index, along with search results for both a single conjunctive query, and a set of queries issued against the (small) index created for user posts in Math Stack Exchange. 

Next, test indexing and retrieval for the formulas, using:

```
make math
```

Again on the terminal, you will see information on indexing, along with results for a single query and a batch query.


After running these tests, you can try passing different flags to `arqmath-test`, and observe the effect (e.g., using `-l none` to prevent stopword removal and stemming).

**The `src/index_arqmath.py` program has been written to make it easy to scan, modify, and reuse.** You are encouraged to do all three for your project!

**Deleting Index Directories** If at some point you want to get rid of your local index directories, issue (**make sure you want this!**):

```
make clean
```
For the test program this is not a big deal, but later on if you reindex you will want to be careful before issuing this command.

## Important Notes

* ARQMath web pages: <https://www.cs.rit.edu/~dprl/ARQMath>
* PyTerrier was **just** updated on April 10, 2022
	* PyTerrer documentation: [PDF](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjdiKau-Yz3AhWJhIkEHTo0BiUQFnoECAcQAQ&url=https%3A%2F%2Fpyterrier.readthedocs.io%2F_%2Fdownloads%2Fen%2Flatest%2Fpdf%2F&usg=AOvVaw0oDx5sV2EGn-xrsJrDLNQn) -- [online](https://pyterrier.readthedocs.io/en/latest/)
	
* The PyTerrier Query Language: [https://github.com/terrier-org/terrier-core/blob/5.x/doc/querylanguage.md](https://github.com/terrier-org/terrier-core/blob/5.x/doc/querylanguage.md). The QL includes operations to support boolean operations (e.g., + requires a keyword to appear (conjunctive), - requires a keyword not to appear), along with Galago-like operations to require tokens to appear in windows, combine scores, etc. that we discussed in class.

* There are two main indices created by PyTerrier, which may hold slightly different information.

	1. The **inverted index** stores keywords in a fast, searchable index
	2. The **metadata index** is what we referred to earlier in class as the 'document index.' This index records document contents for use in generating hit summaries and retrieving information on matched documents that is not provided in the inverted index used for search.

* **PyTerrier is designed to remove punctuation by default** from queries and documents. However, formulas in LaTeX have a *lot* of punctuation. To address this:
	* Punctuation in math strings and input queries are mapped to text tokens (see `src/math_recoding.py`)
	* In the provided test program (see below), the **inverted index** is constructed using tokens for punctuation. However, for readability and to save space, the **metadata index** contains formulas using LaTeX from the original posts. It is possible to change this if desired (see the code).
	* **Modified PyTerrier Query Language for ARQMath.** As a side-effect, the PyTerrier Query Language operators need to be defined differently in query strings. The current code requires the user to use `_pand` for `+` (required/conjunctive) and `_pnot` for `-`, for example. See the `test_retrieval()` function in `sec/index_arqmath.py` for an example.
	*  To retrieve answer posts only, add `_pnot qpost` to your query. The example test program makes use of this, and you can compare results with and without this in the test program runs.
	*  **So far, we have been unable to get the field-based search** to work in the PyTerrier QL (even if fields are capitalized as the PyTerrier error messages suggest doing). Ideally, this would allow us to search a title for a keyword using `TITLE:keyword`.


## Search and Evaluation Using the ARQMath Collection

(TBD) - stay tuned.


## Authors and License

Created for the CSCI 539 course (Information Retrieval) at RIT, April 2022  
**Authors:** Richard Zanibbi (<rxzvcs@rit.edu>) and Behrooz Mansouri (<bm3302@rit.edu>)    
**License:** Mozilla Public License 2.0 (per PyTerrier)
