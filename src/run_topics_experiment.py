################################################################
# play_with_data.py
#
# Test program and experiment data to better understand it
#
# Author:
# Sri Kamal V. Chillarage, Apr 2022
################################################################

# Which experiments to run? #
RUN_BERT=True
RUN_COLBERT=False
#############################

import imp
from index_arqmath import *
from arqmath_topics_qrels import *
import argparse
import pyterrier as pt
from pyterrier.measures import *

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

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

def generate_weighting_experiment(pipelineA, pipelineB, end_pipeline, nameA, nameB, prefix):
    weights = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result_pipelines = []
    result_names = []
    for weightA in weights:
        weightB = 10 - weightA
        result_pipelines.append( ((weightA * pipelineA) + (weightB * pipelineB)) >> end_pipeline )
        result_names.append(f"{prefix}-{nameA}_{weightA}-{nameB}_{weightB}")
    return result_pipelines, result_names

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

    ############################
    ### Setup Search Engines ###
    ############################
    ## Math and text token pipelines
    math_token_pipeline = ""
    text_token_pipeline = "Stopwords,PorterStemmer"

    ## Prime transformer
    prime_transformer = select_assessed_hits(qrels_df, top_k, prime)

    ## Math processor pipeline stage
    math_correct_data = (pt.apply.generic(
                lambda df: df.rename(columns={'docno': 'formulano'}))  # rename columns
                >> pt.apply.generic(
                lambda df: df.rename(columns={'postno': 'docno'}))  # rename columns
                >> pt.apply.generic(
                lambda df: df.drop_duplicates(subset=['docno']))
                )

    ## Raw BM25 engines
    bm25_math_engine = search_engine(math_index, weight_model, MATH_META_FIELDS, token_pipeline=math_token_pipeline) >> math_correct_data % 1000
    bm25_post_engine = search_engine(post_index, weight_model, TEXT_META_FIELDS, token_pipeline=text_token_pipeline) % 1000

    #################
    ### PIPELINES ###
    #################
    experiments = []
    experiment_names = []

    ## Manually set math and post weights
    math_engine_weight = 8
    post_engine_weight = 10 - math_engine_weight

    ## Baseline Experiment
    baseline = bm25_post_engine >> prime_transformer
    experiments.append(baseline)
    experiment_names.append("Baseline")

    ## Experiment 1: BM25 math & post pipeline with linear interpolation
    # following comment out was used to generate experiments to find the correct BM25 weightings for math and posts
    #ex_1_pipelines, ex_1_titles = generate_weighting_experiment(bm25_math_engine, bm25_post_engine, prime_transformer, "math", "post", "BM25")
    experiment_1 = ((bm25_post_engine * post_engine_weight) + (bm25_math_engine * math_engine_weight)) >> prime_transformer
    experiments.append(experiment_1)
    experiment_names.append("BM25 to Linear Interpolation")

    import tokenizers
    
    ## Experiment 2: create math & post pipeline, ColBERT base model re-ranking, then linear interpolation
    if RUN_COLBERT:
        print("Initializing ColBERT base model...")
        import pyterrier_colbert.ranking
        colbert_base_factory = pyterrier_colbert.ranking.ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip", None, None)
        vanilla_colbert_rerank_math_engine = (bm25_math_engine * math_engine_weight) >> colbert_base_factory.text_scorer()
        vanilla_colbert_rerank_post_engine = (bm25_post_engine * post_engine_weight) >> colbert_base_factory.text_scorer()
        experiment_2 = ((vanilla_colbert_rerank_math_engine) + (vanilla_colbert_rerank_post_engine)) >> prime_transformer
        experiments.append(experiment_2)
        experiment_names.append("BM25 to ColBERT to Linear Interpolation")
    else:
        print("########################### ColBERT DISABLED ###########################")
    
    ## Experiment 3: create math & post pipeline, vanilla BERT, then linear interpolation
    if RUN_BERT:
        import onir_pt
        print("Initializing BERT base model...")
        vbert = onir_pt.reranker('vanilla_transformer', 'bert', text_field='origtext', vocab_config={'train': True})
        vanilla_bert_rerank_math_engine = bm25_math_engine >> vbert
        vanilla_bert_rerank_post_engine = bm25_post_engine >> vbert
        experiment_3 = ((vanilla_bert_rerank_math_engine * math_engine_weight) + (vanilla_bert_rerank_post_engine * post_engine_weight)) >> prime_transformer
        experiments.append(experiment_3)
        experiment_names.append("BM25 to VBERT to Linear Interpolation")
    else:
        print("########################### BERT DISABLED ###########################")

    ## Run experiments
    print("Running topics...")
    ndcg_metrics = pt.Experiment(
        experiments,
        query_df,
        qrels_df,
        baseline=0,
        eval_metrics=["ndcg", "mrt"],
        names=experiment_names,
        save_dir="./",
        save_mode="overwrite"
    )
    print("ran ndcg metrics")
    binarized_metrics = pt.Experiment(
        experiments,
        query_df,
        qrels_thresholded,
        baseline=0,
        eval_metrics=["P_10", "map", "mrt"],
        names=experiment_names,
        save_dir="./"
    )
    print("ran binary experiment")
    # Report results at the command line.
    report_results(ndcg_metrics, binarized_metrics, top_k, prime)


main()
