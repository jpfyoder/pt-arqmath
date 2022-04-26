################################################################
# play_with_data.py
#
# Test program and experiment data to better understand it
#
# Author:
# Sri Kamal V. Chillarage, Apr 2022
################################################################

from index_arqmath import *
from arqmath_topics_qrels import *
import argparse
import pyterrier as pt
from pyterrier.measures import *
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import os, sys

# Constants
REL_THRESHOLD=2  # ARQMath convention, treat '2' in 0-3 scale as 'relevant' for binary relevance metrics
MAX_HITS=1000    # TREC / CLEF / NTCIR / FIRE / ARQMath convention (max of 1000 hits per query)


def process_args():
    # Process command line arguments
    parser = argparse.ArgumentParser(description="Retrieve and evaluate results for ARQMath data.")

    parser.add_argument('mathIndexDir', help='Directory containing ARQMath math index')
    parser.add_argument('postIndexDir', help='Directory containing ARQMath post index')
    parser.add_argument('xmlFile', help='ARQMath topics file (XML)')
    parser.add_argument('qrelFile', help='ARQMath qrels file (**needs to correspond to topic file)')
    parser.add_argument('-m', '--model', default="BM25",
                        help="term weight model (default: BM25; TF_IDF + other PyTerrier built-ins are available)")
    parser.add_argument('-k', '--topk', type=int, default=1000,
                        help="select top-k hits (default: 1000)")
    parser.add_argument('-nop', '--noprime', default=False,
                        help="compute non-prime metrics (default: False)", action="store_true")
    parser.add_argument('-l', '--lexicon', help='show lexicon', action="store_true")
    parser.add_argument('-s', '--stats', help="show collection statistics", action="store_true")
    parser.add_argument('-t', '--tokens',
                        help="set tokenization property (Stopwords,PorterStemmer:  no stemming/stopword removal)",
                        default='none')
    parser.add_argument('-d', '--debug', help="include debugging outputs", action="store_true", default='-d')

    args = parser.parse_args()

    return args


def load_topics( file_name ):
    ( num_topics, topics_df ) = read_topic_file( file_name )
    queries = topics_df[ 'query' ]
    qids = topics_df[ 'qid' ]
    query_df = pt.new.queries( queries, qid=qids)

    return ( num_topics, query_df )


def load_qrels_with_binary( file_name ):
    qrels_df = load_qrels( file_name )
    # !! Create thresholded qrels for map' and p'@10
    qrels_thresholded = qrels_df[ qrels_df['label'] >= REL_THRESHOLD ]

    return ( qrels_df, qrels_thresholded )


def load_index( index_dir, lexicon, stats ):
    index_path = index_dir + "/data.properties"
    print("Loading index defined at " + index_path + "...")
    index_ref = pt.IndexRef.of( index_path )
    index = pt.IndexFactory.of( index_ref )

    # If asked, report stats and lexicon
    view_index( index_dir, index, lexicon, stats )

    return index


def report_results( ndcg_metrics, binarized_metrics, top_k, prime ):
    # Make clear what we're using!
    prime_string = ''

    if prime:
        prime_string ="'"
    print("[[ Evaluation  ]]")
    print(" * Top-k hits evaluated: " + str(top_k ) )
    print(" * Prime metrics ('): " + str(prime) )
    print(" * !! Note that ARQMath uses prime metrics for official scores.")
    print("\nResults for nDCG" + prime_string )
    print("----------------------------------------------------------")
    print( ndcg_metrics )

    print("\nResults for binarized relevance : mAP" + prime_string + ", Precision at 10" + prime_string)
    print("----------------------------------------------------------")
    print( binarized_metrics )

    print("\ndone.")


################################################################
# Evaluation
################################################################
# Used to remove unasessed hits in search results for prime (') metrics
# Consider only up to MAX_HITS
def select_assessed_hits( qrel_df, top_k=1000, prime=True ):
    def filter_results( result_df ):
        #result_df.drop_duplicates( subset='docno' )  # esp. important for formula retrieval results
        #result_df_cut = result_df.iloc[0 : MAX_HITS ]
        result_df_cut = result_df.loc[ result_df[ 'rank' ] < top_k ]
        out_results = result_df_cut

        # Prime metrics
        # If 'prime' is true, filter by ( qid, docno ) pairs in the qrel file, so that
        # only assessed hits are included.
        if prime:
            keycols = qrel_df[ ['qid','docno'] ]
            keys = list( keycols.columns.values )
            i1 = result_df_cut.set_index(keys).index
            i2 = qrel_df.set_index(keys).index
            out_results = result_df_cut[ i1.isin(i2) ]

        return out_results

    return pyterrier.apply.generic( filter_results )


def main():
    # Process arguments
    args = process_args()
    # Set pandas display width wider
    pd.set_option('display.max_colwidth', 150)

    if args.tokens == 'none':
        args.tokens = ''

    # Set retrieval and evaluation parameters
    weight_model = args.model
    prime = not args.noprime
    top_k = args.topk

    # Do not forget, or fields are undefined ('None' in error messages)
    print('\n>>> Initializing PyTerrier...')
    if not pt.started():
        pt.init()

    print("\n>>> Starting up ")

    # Collect topics, qrels index
    print("Loading topics (queries)...")
    (num_topics, query_df) = load_topics(args.xmlFile)
    print("    " + str(num_topics) + " topics lodaded.")

    print("Loading qrels...")
    (qrels_df, qrels_thresholded) = load_qrels_with_binary(args.qrelFile)

    print("Loading math index defined at " + args.mathIndexDir + "...")
    math_index = load_index(args.mathIndexDir, args.lexicon, args.stats)

    print("Loading post index defined at " + args.postIndexDir + "...")
    post_index = load_index(args.postIndexDir, args.lexicon, args.stats)

    # print(pt.BatchRetrieve(index).search("Rationals can be the set of continuity of a function question"))  # confirmed index works

    # Report tokenization
    # token_pipeline = index.getProperty("termpipelines")  # does not work.
    # print("Tokenization: " + token_pipeline)

    print("Generating search engine...(" + weight_model + ") with tokenization spec: '" + args.tokens + "')")
    # Compiling example to make it faster (see https://pyterrier.readthedocs.io/en/latest/transformer.html)
    # * Filtering unasessed hits (w. prime_transformer) - also enforces maximum result list length.
    #tokenizer = AutoTokenizer.from_pretrained("vespa-engine/colbert-medium")
    #model = ColBERT.from_pretrained("vespa-engine/colbert-medium")

    print("tokenizer and model created!")

    prime_transformer = select_assessed_hits(qrels_df, top_k, prime)

    # Create BM25 engines for math and posts

    bm25_math_engine = search_engine(math_index, weight_model, MATH_META_FIELDS, token_pipeline="")
    bm25_post_engine = search_engine(post_index, weight_model, TEXT_META_FIELDS, token_pipeline="Stopwords,PorterStemmer")


    # Retraining ColBERT
    #train_ds = pt.datasets.get_dataset('ARQMath_Collection-math-ptindex')
    #train_topics, valid_topics = train_test_split(train_ds.get_topics(), test_size=50, random_state=42) # split into training and validation sets

    #import pyterrier_colbert.ranking
    #colbert_factory = pyterrier_colbert.ranking.ColBERTFactory(
    #"http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip", "arq-math/pt-arqmath/ARQMath_Collection-math-ptindex", None)
    # old ColBERT pipeline
    #bm25_colbert_post_pipe = (pt.BatchRetrieve(post_index, wmodel="BM25") % 100 # get top 100 results using bm25
    #        >> pt.text.get_text(train_ds, 'text') # fetch the document text
    #        >> colbert_factory) # apply neural re-ranker

    # Manually set weights
    math_pipeline_weight = 0.75
    post_pipeline_weight = 0.25

    ## Baseline Experiment
    baseline = bm25_post_engine >> prime_transformer

    ## Experiment 1: create math & post pipeline, no ColBERT, then linear interpolation
    bm25_math_pipeline_base = (bm25_math_engine >> pt.apply.generic(
                lambda df: df.rename(columns={'docno': 'formulano'}))  # rename columns
                >> pt.apply.generic(
                lambda df: df.rename(columns={'postno': 'docno'}))  # rename columns
                >> pt.apply.generic(
                lambda df: df.drop_duplicates(subset=['docno']))
                )
    bm25_post_pipeline_base = bm25_post_engine
    experiment_1 = ((math_pipeline_weight * bm25_math_pipeline_base) + (post_pipeline_weight * bm25_post_pipeline_base)) >> prime_transformer
    
    ## Experiment 2: create math & post pipeline, ColBERT base model re-ranking, then linear interpolation

    ## Experiment 3: create math & post pipeline, ColBERT re-trained model re-ranking, then linear interpolation

    print("Running topics...")
    ndcg_metrics = pt.Experiment(
        [baseline, experiment_1],
        query_df,
        qrels_df,
        baseline=0,
        eval_metrics=["ndcg", "mrt"],
        names=["Baseline", "Experiment 1"],
        save_dir="./",
        save_mode="overwrite"
    )
    print("ran ndcg metrics")
    binarized_metrics = pt.Experiment(
        [baseline, experiment_1],
        query_df,
        qrels_thresholded,
        baseline=0,
        eval_metrics=["P_10", "map", "mrt"],
        names=["Baseline", "Experiment 1"],
        save_dir="./"
    )
    print("ran binary experiment")
    # Report results at the command line.
    report_results(ndcg_metrics, binarized_metrics, top_k, prime)


main()
