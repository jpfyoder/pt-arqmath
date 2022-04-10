################################################################
# index-arqmath.py
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
import os
import argparse
from tqdm import tqdm

################################################################
# Index creation and properties
################################################################
def rewrite_math_tags( soup ):
    formulaTags = soup('span')
    formula_ids = [ node['id'] for node in formulaTags ]
    for tag in formulaTags:
        tag.name = 'math'
        del tag['class']
        #del tag['id']

    return ( formulaTags, formula_ids )

def generate_XML_post_docs(file_name, formula_index=False, debug_out=False ):
    # **Warning: tag attribute names converted to lower case by default in BSoup
    tagsToRemove = ['p','a','body','html']

    print(">> Reading: ", file_name )
    with open(file_name) as xml_file:
        soup = bsoup(xml_file, 'lxml')
        rows = soup('row')

        for row in tqdm( rows ):
            # Parse post body and title content as HTML & get formulas
            # Document number in collection, user votes
            docno = row['id'] 
            votes = row['score']

            # Parent post for answers ('none' for questions)
            parentno = 'none'
            if row[ 'posttypeid' ] == '2':  
                parentno = row['parentid'] 

            # Title formulas - apply soup to recover HTML structure from attribute field value
            title_soup = bsoup( html.unescape( row.get('title','') ), 'lxml' )
            ( title_formulas, title_formula_ids ) = rewrite_math_tags( title_soup )
            for tag in title_soup( tagsToRemove ):
                tag.unwrap()

            # Body formulas and simplification - again, apply soup to construct Tag tree w. bsoup
            body_soup = bsoup( html.unescape( row['body'] ), 'lxml' )
            ( body_formulas , formula_ids )= rewrite_math_tags( body_soup )
            # Remove unwanted <p>, <body>, <html> tags
            for tag in body_soup( tagsToRemove ):
                tag.unwrap()
    
            # Combine title and body formulas
            all_formulas = title_formulas + body_formulas
            all_formula_ids = title_formula_ids + formula_ids

            if formula_index:
                ## Formula index entries   ##
                #  One output per formula
                for math_tag in all_formulas:
                    yield { 'docno':     math_tag['id'],
                            'text':      math_tag.get_text(),
                            'postno':    docno,
                            'parentno' : parentno
                        }
            else:
                ## Post text index entries ##
                # Remove formula ids from title and body
                for math_tag in all_formulas:
                    del math_tag['id']

                # Generate strings for title, post body, and tags
                title_text = str( title_soup )
                modified_post_text = str( body_soup )
                tag_text = row.get('tags', '').replace('<','').replace('>',', ').replace('-',' ')

                # DEBUG: Show main text field entries.
                if debug_out:
                    print('\nDOCNO: ',docno,'\nTITLE: ',title_text,'\nBODY: ',modified_post_text,'\nTAGS: ',tag_text)

                # Note: the formula ids are stored in a string currently.
                # Concatenate post and tag text
                yield { 'docno' :   docno,
                        'title' :   title_text,
                        'text' :    modified_post_text,
                        'tags' :    tag_text,
                        'mathnos' : all_formula_ids,
                        'parentno': parentno,
                        'votes' :   votes
                    }


def create_XML_index( file, indexDir, token_pipeline="Stopwords,PorterStemmer", formulas=False, debug=False):
    # Construct an index
    # Post meta (document index) fields
    meta_fields=['docno','title', 'text', 'tags', 'votes', 'parentno', 'mathnos' ]
    meta_sizes=[16, 256, 4096, 128, 8, 20, 20]
    field_names= [ 'title', 'text', 'tags', 'parentno' ]

    if formulas:
        # Formula index fields: redefine
        meta_fields=['docno','text','postno','parentno']
        meta_sizes=[20, 1024, 20, 20]
        field_names=[ 'text', 'parentno' ]

    indexer = pt.IterDictIndexer( indexDir, 
            meta=meta_fields,
            meta_lengths=meta_sizes,
            overwrite=True )
    indexer.setProperty( "termpipelines", token_pipeline )
    
    index_ref = indexer.index( generate_XML_post_docs( file, formula_index=formulas, debug_out=debug ), fields=field_names )
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
        print('[ ' + indexName + ': Details ]')
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
    return engine.search( query )

# Run a list of queries
def batch_query( engine, query_list ):
    column_names=[ "qid", "query" ]
    
    query_count = len(query_list)
    qid_list = [ str(x) for x in range(1, query_count + 1) ]

    query_pairs = list( zip( qid_list, query_list ) )
    queries = pd.DataFrame( query_pairs, columns=column_names )

    return engine( queries )

def test_retrieval( post_index, math_index, model, tokens, debug=False ):

    if post_index != None:
        print("[ Testing post index retrieval ]")
        
        posts_engine = search_engine( post_index, 
                model, 
                metadata_keys = [ 'title' ],
                #[ 'docno', 'text', 'mathnos', 'parentno' ], 
                token_pipeline=tokens )
        
        print( query( posts_engine, '+simplified +proof' ) )
        print('')
        print( batch_query( posts_engine, ['simplified proof', 'proof' ] )['title'] )
        print('')
        
        print( query( posts_engine, '+dungeons' ) )
        print('')
    
    if math_index != None:
        print("[ Testing math index retrieval ]")
        
        math_engine = search_engine( math_index, model, ['docno', 'text', 'postno', 'parentno' ], token_pipeline=tokens )
        print( query( math_engine, '+sqrt +2' ) )
        print( '' )
        print( batch_query( math_engine, [ 'sqrt 2', '2' ] ) )

        print( '' )
    print( 'Test complete.' )


################################################################
# Main program
################################################################
def process_args():
    # Process command line arguments
    parser = argparse.ArgumentParser(description="Indexing tool for ARQMath data.")

    parser.add_argument('xmlFile', help='ARQMath XML file to index')
    xgroup = parser.add_mutually_exclusive_group(required=False)
    xgroup.add_argument('-m', '--math', help='create only the math index', action="store_true" )
    xgroup.add_argument('-mp', '--mathpost', help='create math and post indices', action="store_true")
    parser.add_argument('-l', '--lexicon', help='show lexicon', action="store_true" )
    parser.add_argument('-s', '--stats', help="show collection statistics", action="store_true" )
    parser.add_argument('-t', '--tokens', help="set tokenization property ('':  no stemming/stopword removal)", default='Stopwords,PorterStemmer' )
    parser.add_argument('-d', '--debug', help="include debugging outputs", action="store_true" )
    
    args = parser.parse_args()
    return args


def main():
    # Process arguments
    args = process_args()
    ( indexDir, _ ) = os.path.splitext( os.path.basename( args.xmlFile ) )
    # Set pandas display width wider
    pd.set_option('display.max_colwidth', 150)

    # Start PyTerrier -- many Java classes unavailable until this is complete
    if not pt.started():
        pt.init()

    # Initialize indices as non-existent
    post_index = None
    math_index = None
    
    # Post index construction
    # Store post text and ids for formulas in each post in the 'meta' (document) index
    if not args.math or args.mathpost:
        post_index = create_XML_index( args.xmlFile, "./" + indexDir + "-post-ptindex", token_pipeline=args.tokens, debug=args.debug )
        view_index( "Post Index", post_index, args.lexicon, args.stats )

    # Formula index construction
    # Store formula text (LaTeX) and formula ids, along with source post id for each formula
    if args.math or args.mathpost:
        math_index = create_XML_index( args.xmlFile, "./" + indexDir + "-math-ptindex", formulas=True, token_pipeline=args.tokens, debug=args.debug )
        view_index( "Math Index", math_index, args.lexicon, args.stats )

    # Retrieval test
    test_retrieval( post_index, math_index, 'BM25', args.tokens, debug=args.debug )    

if __name__ == "__main__":
    main()
