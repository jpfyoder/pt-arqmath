################################################################
# run_topics.py
#
# Example program running BM25 on ARQMath 
# posts and topics with textualized formulas
#
# Author:
# R. Zanibbi, Apr 2022
################################################################

from index_arqmath import *
from arqmath_topics_qrels import *
import argparse
import pyterrier as pt
from pyterrier.measures import *
import os

# Constants
REL_THRESHOLD=2  # ARQMath convention, treat '2' in 0-3 scale as 'relevant' for binary relevance metrics
MAX_HITS=1000    # TREC / CLEF / NTCIR / FIRE / ARQMath convention (max of 1000 hits per query)

def process_args():
    # Process command line arguments
    parser = argparse.ArgumentParser(description="Retrieve and evaluate results for ARQMath data.")

    parser.add_argument('indexDir', help='Directory containing ARQMath index')
    parser.add_argument('xmlFile', help='ARQMath topics file (XML)')
    parser.add_argument('qrelFile', help='ARQMath qrels file (**needs to correspond to topic file)')
    args = parser.parse_args()
    
    return args

# Used to remove unasessed hits in search results for prime (') metrics
# Consider only up to MAX_HITS
def select_assessed_hits( qrel_df ):
    # Use Pandas operations to select columns with shared 'qid' and 'docno' in a row
    # (need to consider query, not just document identifier)
    def filter_results( result_df ):
        result_df_cut = result_df.iloc[0 : MAX_HITS ]

        # Filter by ( qid, docno ) pairs in the qrel file.
        keycols = qrel_df[ ['qid','docno'] ]
        keys = list( keycols.columns.values )
        i1 = result_df_cut.set_index(keys).index
        i2 = qrel_df.set_index(keys).index

        return result_df_cut[ i1.isin(i2) ]

    return pyterrier.apply.generic( filter_results )

def main():
    # Process arguments
    args = process_args()
    # Set pandas display width wider
    pd.set_option('display.max_colwidth', 150)

    # Do not forget, or fields are undefined ('None' in error messages)
    print('\n>>> Initializing PyTerrier...')
    if not pt.started():
        pt.init()

    print("\n>>> Starting up ")
    # Collect topics, qrels index
    print("Loading topics (queries)...")
    ( num_topics, topics_df ) = read_topic_file( args.xmlFile )
    queries = topics_df[ 'query' ]
    qids = topics_df[ 'qid' ]
    query_df = pt.new.queries( queries, qid=qids)
    print("    " + str(num_topics) + " topics lodaded.")

    print("Loading qrels...")
    qrels_df = load_qrels( args.qrelFile )
    # !! Create thresholded qrels for map' and p'@10
    qrels_thresholded = qrels_df[ qrels_df['label'] >= REL_THRESHOLD ]

    # Load index
    index_path = args.indexDir + "/data.properties" 
    print("Loading index defined at " + index_path + "...")
    index_ref = pt.IndexRef.of( index_path )
    index = pt.IndexFactory.of( index_ref )

    print("Generating search engine...(BM25 with stop word removal and Porter stemming)")
    # Compiling example to make it faster (see https://pyterrier.readthedocs.io/en/latest/transformer.html)
    # * Filtering unasessed hits (w. prime_transformer) - also enforces maximum result list length.
    prime_transformer = select_assessed_hits( qrels_df )
    bm25_engine = search_engine( index, 'BM25', TEXT_META_INDEX_FIELDS ) >> prime_transformer 

    # First pass: two runs on retrieval, one for ndcg', one for binarized metrics
    # Saves results to current directory in file BM25.res.gz this prevents running
    # the retrieval pipeline in the second call.
    #
    # **See warnings about accidentally reusing results and getting incorrect results here: 
    #   https://pyterrier.readthedocs.io/en/latest/experiments.html?highlight=experiments
    #  ('overwrite' used to create new results for each run of this program for the first retrieval)
    print("Running topics...")
    ndcg_metrics = pt.Experiment(
        [ bm25_engine ],
        query_df,
        qrels_df,
        eval_metrics=[ "ndcg", "mrt" ],
        names=["BM25"],
        save_dir="./",
        save_mode="overwrite"
        )

    # This does not run the pipeline again, but does compute different metrics used the save
    # search results.
    binarized_metrics = pt.Experiment(
        [ bm25_engine ],
        query_df,
        qrels_thresholded,
        eval_metrics=[ "P_10", "map", "mrt" ],
        names=["BM25"],
        save_dir="./"
        )


    print("\nResults for nDCG' : used to rank ARQMath systems")
    print("----------------------------------------------------------")
    print( ndcg_metrics )

    print("\nResults for binarized relevance : mAP', Precision' at 10")
    print("----------------------------------------------------------")
    print( binarized_metrics )

    print("\ndone.")


main()

