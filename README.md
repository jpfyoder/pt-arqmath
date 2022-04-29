## Group 1 RIT CSCI-539 Information Retrieval Modifications to Code

Code can be run by installing requirements as normal, just run the `make` command. To run our experiment, do the following:

- `make-experiment-short` will run the short experiment (subset of the 2020 as provided in class)
- `make-experiment-2020` will run the experiment over the 2020 collection.
- `make-experiment-2021` will run the experiment over the 2021 collection.

All of our code is in the `run_topics_experiment.py` file.

**Note about BERT and ColBERT**: The BERT and ColBERT implementations used in pyterrier require different versions of the `transformers` library to run properly. To fix this, the code will only run one of the experiments at a time, you must switch libraries to run BERT.

*Global variables* at the top of the `run_topics_experiment.py` file control weather BERT or ColBERT is being run. By default, it is BERT. To swicth to ColBERT you must update the `RUN_COLBERT` variable to `True` and `RUN_BERT` variable to `False`.

- 

As always, `make-experiment-*` will run the respective experiments.

## PyTerrier Framework for ARQMath Task 1

`pt-arqmath` was created for the Information Retrieval course at the Rochester Institute of Technology in Spring 2022. PyTerrier provides a flexible framework for building, running, and comparing a variety of different search engines, including neural retrieval models. 

This code is provided to make getting started with ARQMath Task 1 (Answer Retrieval) a bit easier. The rest of this document covers:

* Installation of the git package 
* Getting started: Indexing
* Search and evaluation using the ARQMath collection
* Important notes (especially for CSCI 539 students)
* Acknowledgements and Author information


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


## Getting Started: Indexing

Some quick indexing and retrieval tests are provided by the `arqmath-index` bash script. The script has flags you can modify, for example to return index statistics, the lexicon produced after tokenization, whether to produce an index for posts/formulas/both, and a flag to control tokenization by PyTerrier  (e.g., stemming and stopword removal).

If you issue `./arqmath-index` without arguments, you should see the following:

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


After running these tests, you can try passing different flags to `arqmath-index`, and observe the effect (e.g., using `-l none` to prevent stopword removal and stemming).

**The `src/index_arqmath.py` program has been written to make it easy to scan, modify, and reuse.** You are encouraged to do all three for your project!

**Deleting Index Directories** If at some point you want to get rid of your local index directories, issue (**make sure you want this!**):

```
make delete-indices
```
For the test program this is not a big deal, but later on if you reindex you will want to be careful before issuing this command.




## Search and Evaluation Using the ARQMath Collection

To do a quick test of the evaluation framework, we have created a preliminary BM25 model that you can run quickly on a couple of queries by issuing:

```
make eval
```
which runs two queries, that don't do particularly well (!). This is partly because currently only the topic question titles are used in the search queries. The other topic fields are read and stored by the `read_topic_file` function in `src/arqmath_topics_qrels.py` that read topic filess; note that the `text` field defining queries may be easily modified.

You can look at the (shortened) topics file in `test/2020_topics_task1_short.xml` for an example of the topics file format.

To run this BM25 model over *all* topics from ARQMath-1 (2020), issue:

```
./run-topics-2020
```
and to run the model over all topics from ARQMath-2 (2021), issue:

```
./run-topics-2021
```

### Notes on the Evaluation Protocol for ARQMath Task 1

* **Per TREC-based evaluation protocol conventions, only the top 1000 hits from each search result should be passed on for evaluation.** The provided code does this already.
* **Metrics are computed in 'prime' form.** For prime versions of metrics, only documents with relevance ratings in qrel files are used in evaluation. For example, P'@5 requires first removing documents from the ranked results that have not been evaluated (i.e., are not included in the qrels), and then computing precision for the top 5 remaining documents. 
*  **Prime metrics were adopted in ARQMath to allow systems run outside of the official lab runs to be fairly compared to participants.** By default, `trec_eval` scores unevaluated documents as non-relevant, which can drive the scores of systems that did not contribute to the assessment pools down, even if they are able to recover relevant documents that are not included in qrels. Prime metrics avoid this, by comparing systems using a fixed set of assessed documents.
*  The function `select_assessed_hits` in `src/run_topics.py` implements both the cut off at 1000 hits, and the removal of documents not included in the qrels.

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


## Acknowledgement

We thank Craig Macdonald, Nicola Tonellotto, and the many other PyTerrier and Terrier contributors for making this helpful framework available. 

If you use this system, you should cite their paper shown below. An arXiv version of the paper may be found at this [link](https://arxiv.org/abs/2007.14271).

```bibtex
@inproceedings{pyterrier2020ictir,
    author = {Craig Macdonald and Nicola Tonellotto},
    title = {Declarative Experimentation in Information Retrieval using PyTerrier},
    booktitle = {Proceedings of ICTIR 2020},
    year = {2020}
}
```

## Authors and License

Created for the CSCI 539 course (Information Retrieval) at RIT, April 2022  
**Authors:** Richard Zanibbi (<rxzvcs@rit.edu>) and Behrooz Mansouri (<bm3302@rit.edu>)    
**License:** Mozilla Public License 2.0 (per PyTerrier)
