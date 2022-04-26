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
import os, sys

# Constants
REL_THRESHOLD=2  # ARQMath convention, treat '2' in 0-3 scale as 'relevant' for binary relevance metrics
MAX_HITS=1000    # TREC / CLEF / NTCIR / FIRE / ARQMath convention (max of 1000 hits per query)


def process_args():
    # Process command line arguments
    parser = argparse.ArgumentParser(description="Retrieve and evaluate results for ARQMath data.")

    parser.add_argument('indexDir', help='Directory containing ARQMath index')
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

    print("Loading index defined at " + args.indexDir + "...")
    index = load_index(args.indexDir, args.lexicon, args.stats)

    # Report tokenization
    # token_pipeline = index_ref.getProperty("termpipelines")  # does not work.
    # print("Tokenization: " + token_pipeline)

    print("Generating search engine...(" + weight_model + ") with tokenization spec: '" + args.tokens + "')")
    # Compiling example to make it faster (see https://pyterrier.readthedocs.io/en/latest/transformer.html)
    # * Filtering unasessed hits (w. prime_transformer) - also enforces maximum result list length.
    prime_transformer = select_assessed_hits(qrels_df, top_k, prime)
    bm25_engine = search_engine(index, weight_model, MATH_META_FIELDS, token_pipeline=args.tokens)
    bm25_pipeline = bm25_engine >> prime_transformer

    import pyterrier_colbert.ranking
    colbert_factory = pyterrier_colbert.ranking.ColBERTFactory(
    "http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip", None, None)
    bm25_colbert_pipe = bm25_engine >> colbert_factory.text_scorer() >> prime_transformer

    print(index.getCollectionStatistics().toString())
    print(index.getMetaIndex().getKeys())
    print(query_df)
    print(qrels_df)
    print(top_k)
    print(prime)
    print(qrels_thresholded)

    print("Running topics...")
    ndcg_metrics = pt.Experiment(
        [bm25_pipeline, bm25_colbert_pipe],
        query_df,
        qrels_df,
        eval_metrics=["ndcg", "mrt"],
        names=[weight_model, "BM25 to ColBERT"],
        save_dir="./",
        save_mode="overwrite"
    )
    print("ran ndcg metrics")
    binarized_metrics = pt.Experiment(
        [bm25_engine, bm25_colbert_pipe],
        query_df,
        qrels_thresholded,
        eval_metrics=["P_10", "map", "mrt"],
        names=[weight_model, "BM25 to ColBERT"],
        save_dir="./"
    )
    print("ran binary experiment")
    # Report results at the command line.
    report_results(ndcg_metrics, binarized_metrics, top_k, prime)






main()
