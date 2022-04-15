################################################################
# index_arqmath.py
#
# PyTerrier-based Python program for indexing ARQMath data
#
# Author:
# R. Zanibbi, April 2022 (CSCI 539, Information Retrieval, RIT)
################################################################

import pyterrier as pt
import pandas as pd
from bs4 import BeautifulSoup as bsoup
import html
import os, glob
import argparse
from tqdm import tqdm

from math_recoding import *
from utils import *

# Constants for indexing
# **Warning: tag names converted to lower case by default in BSoup (e.g., <P> -> <p>)
TAGS_TO_REMOVE = ['p','a','body','html','question','head','title']

TEXT_RETRIEVAL_FIELDS = [ 'title', 'text', 'tags', 'parentno' ]
TEXT_META_FIELDS = ['docno','title', 'text', 'origtext', 'tags', 'votes', 'parentno', 'mathnos' ]
TEXT_META_SIZES = [ 16, 256, 4096, 4096, 128, 8, 20, 20 ]

MATH_RETRIEVAL_FIELDS = [ 'text', 'parentno' ]
MATH_META_FIELDS = [ 'docno', 'text', 'origtext','postno','parentno']
MATH_META_SIZES = [ 20, 1024, 1024, 20, 20]

################################################################
# Index creation and properties
################################################################
def remove_tags( soup_node, tag_list ):
    for tag in soup_node( tag_list ):
        tag.unwrap()

def rewrite_math_tags( soup ):
    # Skip span tags without id's (i.e., formulas without identifiers)
    formulaTags = [ node for node in soup('span') if node.has_attr('id') ]
    formula_ids = [ node['id'] for node in formulaTags if node.has_attr('id') ]
    
    #dbshow( "soup", soup )
    #dbshow( "soup('span')", soup('span') )
    #dbshow( "formulaTags", formulaTags )
    #pshow( "formula_ids", formula_ids )

    for tag in formulaTags:
        tag.name = 'math'
        del tag['class']
        #del tag['id']

    return ( formulaTags, formula_ids )

def print_formula_record(math_tag, tokenized_formula, docno, parentno ):
    print('\nDOCNO:',math_tag['id'],'\nTEXT:',tokenized_formula,'\nORIGTEXT:',math_tag.get_text(),'\nPOSTNO:',docno,
                                    '\nPARENTNO',parentno)

def print_post_record( docno, title_text, modified_post_text, indexed_body, 
        tag_text, all_formula_ids, parentno, votes):
    print('\nDOCNO: ',docno,'\nTITLE: ',title_text,'\nBODY: ',modified_post_text,
        '\nTEXT (SEARCHABLE):',indexed_body,'\nTAGS: ',tag_text,'\nMATHNOS:',all_formula_ids,
        '\nPARENTNO:',parentno,'\nVOTES:',votes)

def generate_XML_post_docs(file_name_list, formula_index=False, debug_out=False ):
    for file_name in file_name_list:
        print(">> Reading File: ", file_name )
        with open(file_name) as xml_file:
            soup = bsoup(xml_file, 'lxml')
            rows = soup('row')

            for row in tqdm( rows ):
                # Parse post body and title content as HTML & get formulas
                # Document number in collection, user votes
                docno = row['id'] 
                votes = row['score']

                # Parent post for answers ('none' for questions)
                parentno = 'qpost'
                if row[ 'posttypeid' ] == '2':  
                    parentno = row['parentid'] 

                # Title formulas - apply soup to recover HTML structure from attribute field value
                title_soup = bsoup( html.unescape( row.get('title','') ), 'lxml' )
                remove_tags( title_soup, TAGS_TO_REMOVE )
                ( title_formulas, title_formula_ids ) = rewrite_math_tags( title_soup )

                # Body formulas and simplification - again, apply soup to construct Tag tree w. bsoup
                body_soup = bsoup( html.unescape( row.get('body','') ), 'lxml' )
                remove_tags( body_soup, TAGS_TO_REMOVE )
                ( body_formulas, formula_ids )= rewrite_math_tags( body_soup )

                # Remove tags that we do not want to search.
        
                # Combine title and body formulas
                all_formulas = title_formulas + body_formulas
                all_formula_ids = title_formula_ids + formula_ids

                if formula_index:
                    ## Formula index entries   
                    #  One output per formula
                    for math_tag in all_formulas:
                        tokenized_formula = rewrite_symbols( math_tag.get_text(), latex_symbol_map )

                        # Skip empty formulas
                        if tokenized_formula.isspace():
                            print('!!! WARNING: Empty "text" field for retrieval')
                            print_formula_record(math_tag, tokenized_formula, docno, parentno )
                            continue
                        
                        elif debug_out:
                            print_formula_record( math_tag, tokenized_formula, docno, parentno )

                        yield { 'docno':     math_tag['id'],
                                'text':      tokenized_formula,
                                'origtext':  math_tag.get_text(),
                                'postno':    docno,
                                'parentno' : parentno
                            }
                else:
                    ## Post text index entries ##
                    # Remove formula ids from title and body
                    #for math_tag in all_formulas:
                    #    del math_tag['id']

                    # Generate strings for title, post body, and tags
                    title_text = str( title_soup )
                    modified_post_text = str( body_soup )
                    indexed_body = translate_latex( modified_post_text )
                    tag_text = row.get('tags', '').replace('<','').replace('>',', ').replace('-',' ')

                    # Skip empty posts with empty content
                    if indexed_body.isspace():
                        print('!!! WARNING: Empty "text" field for retrieval')
                        print_post_record( docno, title_text, modified_post_text, indexed_body, 
                            tag_text, all_formula_ids, parentno, votes)
                        continue

                    elif debug_out:
                        print_post_record( docno, title_text, modified_post_text, indexed_body, 
                            tag_text, all_formula_ids, parentno, votes)

                    # Note: the formula ids are stored in a string currently.
                    # Concatenate post and tag text
                    # NOTE: representation for search is tokenized differently than meta/document index version for viewing hits
                    yield { 'docno' :   docno,
                            'title' :   title_text,
                            'text' :    indexed_body,
                            'origtext': modified_post_text,
                            'tags' :    tag_text,
                            'mathnos' : all_formula_ids,
                            'parentno': parentno,
                            'votes' :   votes
                        }


def create_XML_index( file_list, indexName, token_pipeline="Stopwords,PorterStemmer", formulas=False, debug=False):
    # Storing processed text AND original text in meta index, docs, to support neural reranking with keywords, and 
    # viewing original posts
    ( meta_fields, meta_sizes ) = ( TEXT_META_FIELDS, TEXT_META_SIZES )
    field_names= TEXT_RETRIEVAL_FIELDS

    if formulas:
        ( meta_fields, meta_sizes ) = ( MATH_META_FIELDS, MATH_META_SIZES )
        field_names= MATH_RETRIEVAL_FIELDS

    indexer = pt.IterDictIndexer( 
            indexName, 
            meta=meta_fields,
            meta_lengths=meta_sizes,
            overwrite=True )

    indexer.setProperty( "termpipelines", token_pipeline )
    index_ref = indexer.index( generate_XML_post_docs( file_list, formula_index=formulas, debug_out=debug ), fields=field_names )
    
    return pt.IndexFactory.of( index_ref )

## Visualization routines

def show_tokens( index ):
    # Show lexicon entries
    for kv in index.getLexicon():
        print("%s :    %s" % (kv.getKey(), kv.getValue().toString()) )    

def show_index_stats( index ):
    print( index.getCollectionStatistics().toString() )

def view_index( indexName, index, view_tokens, view_stats ):
    if view_tokens or view_stats:
        print('\n[ ' + indexName + ': Details ]')
        if view_stats:
            show_index_stats( index )
        if view_tokens:
            print('Lexicon for ' + indexName + ':')
            show_tokens( index )
            print('')

################################################################
# Search engine construction and search
################################################################
def search_engine( index, 
        model, 
        metadata_keys=[], 
        token_pipeline="Stopwords,PorterStemmer" ):
    return pt.BatchRetrieve( index, wmodel=model, 
            properties={ "termpipelines" : token_pipeline }, 
            metadata = metadata_keys )

# Run a single query
def query( engine, query ):
    return engine.search( translate_query( query ) )

# Run a list of queries
def batch_query( engine, query_list ):
    column_names=[ "qid", "query" ]
    
    query_count = len(query_list)
    qid_list = [ str(x) for x in range(1, query_count + 1) ]

    # Map TeX characters and ARQMath-version query ops (e.g., '_pand' -> '+')
    print( query_list )
    rewritten_query_list = translate_qlist( query_list )
    
    query_pairs = list( zip( qid_list, rewritten_query_list ) )
    queries = pd.DataFrame( query_pairs, columns=column_names )

    return engine( queries )

def verbose_hit_summary( result, math_index=False ):

    result.reset_index()
    for ( index, row ) in result.iterrows():
        #print("KEYS: " + str( row.keys() ) )
        print('QUERY (' + row['qid'] + '): ', row['query'])
        score = "{:.2f}".format( row['score'] )
        
        print('RANK:', index, 'Score:', score)
        if not math_index:
            # Post document
            print('Docid:',row['docid'], 'Post-no:', row['docno'], 'Parent-no:',row['parentno'],'Votes:', row['votes'] )
            if row['parentno'] == 'qpost':
                print('QUESTION TITLE:', row['title'])
            else:
                print('ANSWER')
        else:
            # Formula document
            print('Docid:',row['docid'], 'Formula-no:', row['docno'],  'Post-no:', row['postno'], 'Parent-no:',row['parentno'])

        # Show original text before token mapping
        print('TEXT:\n', row['text'])
        print('ORIGTEXT:\n',row['origtext'])

        # Provide tags, formula id's for posts
        if not math_index:
            print('TAGS:',row['tags'])
            print('FORMULA IDS:',row['mathnos'])
        
        print('')

def show_result( result, field_names=[], show_table=True, show_hits=False, math=False ):
    print("\n__ SEARCH RESULTS _________________________\n")

    if show_table:
        if field_names == []:
            print( result, '\n' )
        else:
            print( result[ field_names ], '\n' )

    if show_hits:
        verbose_hit_summary( result, math_index=math )

def test_retrieval( k, post_index, math_index, model, tokens, debug=False ):
    titles = [ 'qid', 'docid', 'docno', 'title', 'rank', 'score', 'query' ]
    tags = [ 'qid', 'docid', 'docno', 'tags', 'rank', 'score' , 'query']
    votes = [ 'qid', 'docid', 'docno', 'votes', 'rank', 'score' , 'query']
    parentno = [ 'qid', 'docid', 'docno', 'parentno', 'rank', 'score' , 'query']
    mathnos  = [ 'qid', 'docid', 'docno', 'mathnos', 'rank', 'score', 'query' ]

    if post_index != None:
        print("[ Testing post index retrieval ]")
        
        # Return top k results (% k)
        posts_engine = search_engine( post_index, 
                model, 
                metadata_keys=['docno','title', 'text', 'origtext', 'tags', 'votes', 'parentno', 'mathnos' ],
                token_pipeline=tokens ) % k
        
        result = query( posts_engine, '_pand simplified _pand proof' )
        show_result( result, mathnos, show_hits=True )
        # Added 'writing' to test matching tags, 'mean' in title field  for post number '1'
        show_result( batch_query( posts_engine, [
            'simplified proof', 
            'proof', 
            'writing', 
            'mean', 
            'qpost', 
            'proof _pnot qpost' 
            # 'man +TITLE:{intuition}'  # Trouble restricting to fields (?)
            ] ), parentno, show_hits=True )
    
    if math_index != None:
        print("[ Testing math index retrieval ]")
        
        # Return top k results (% k)
        math_engine = search_engine( math_index, model, ['docno', 'text', 'origtext','postno', 'parentno' ], token_pipeline=tokens ) % k
        show_result( query( math_engine, '_pand sqrt _pand 2' ), show_hits=True, math=True )
        show_result( batch_query( math_engine, [ 'sqrt 2', '2' ] ), show_hits=True, math=True )
        show_result( batch_query( math_engine, [ 'sqrt 2 _pnot qpost' ] ), show_hits=True, math=True )

    print( 'Test complete.' )


################################################################
# Main program
################################################################
def process_args():
    # Process command line arguments
    parser = argparse.ArgumentParser(description="Indexing tool for ARQMath data.")

    parser.add_argument('xmlFile', help='ARQMath XML file or directory of XML files to index')
    xgroup = parser.add_mutually_exclusive_group(required=False)
    xgroup.add_argument('-m', '--math', help='create only the math index', action="store_true" )
    xgroup.add_argument('-mp', '--mathpost', help='create math and post indices', action="store_true")
    parser.add_argument('-l', '--lexicon', help='show lexicon', action="store_true" )
    parser.add_argument('-s', '--stats', help="show collection statistics", action="store_true" )
    parser.add_argument('-t', '--tokens', help="set tokenization property (none:  no stemming/stopword removal)", 
            default='Stopwords,PorterStemmer' )
    parser.add_argument('-n', '--notest', help="skip retrieval tests after indexing", action="store_true" )
    parser.add_argument('-d', '--debug', help="include debugging outputs", action="store_true" )
    
    args = parser.parse_args()
    return args


def main():
    # Process arguments
    args = process_args()
    ( indexName, _ ) = os.path.splitext( os.path.basename( args.xmlFile ) )
    # Set pandas display width wider
    pd.set_option('display.max_colwidth', 150)

    if args.tokens == 'none':
        args.tokens = ''

    in_file_list = [ args.xmlFile ]
    if os.path.isdir( args.xmlFile ):
        # Get directory, keep only xml files.
        dir_files = os.listdir( args.xmlFile )
        in_file_list = [ (args.xmlFile + "/" + file).replace("//","/")
                for file in dir_files if file.endswith('.xml') ]

    # Start PyTerrier -- many Java classes unavailable until this is complete
    print('\n>>> Initializing PyTerrier...')
    if not pt.started():
        pt.init()

    print('\n>>> Indexing...')
    # Initialize indices as non-existent
    post_index = None
    math_index = None
    
    # Post index construction
    # Store post text and ids for formulas in each post in the 'meta' (document) index
    if not args.math or args.mathpost:
        post_index = create_XML_index(
            in_file_list, "./" + indexName + "-post-ptindex", 
            token_pipeline=args.tokens, debug=args.debug )
        view_index( "Post Index", post_index, args.lexicon, args.stats )

    # Formula index construction
    # Store formula text (LaTeX) and formula ids, along with source post id for each formula
    if args.math or args.mathpost:
        math_index = create_XML_index( 
            in_file_list, "./" + indexName + "-math-ptindex", formulas=True, 
            token_pipeline=args.tokens, debug=args.debug )
        view_index( "Math Index", math_index, args.lexicon, args.stats )

    print('>>> Indexing complete.\n')

    # Retrieval test
    # Top k
    k = 5
    if not args.notest:
        test_retrieval( k, post_index, math_index, 'BM25', args.tokens, debug=args.debug )    

if __name__ == "__main__":
    main()
