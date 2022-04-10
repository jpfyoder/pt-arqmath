################################################################
# 
#
# Author:
# R. Zanibbi, April 2022 (CSCI 539, Information Retrieval, RIT)
################################################################

import pyterrier as pt
import pandas as pd
from bs4 import BeautifulSoup as bsoup
import html
import argparse

################################################################
# Generators for items to index (posts and formulas)
################################################################
def generate_XML_post_docs(file_name, formula_index=False):
    # **Warning: tag attribute names converted to lower case by default in BSoup
    with open(file_name) as xml_file:
        soup = bsoup(xml_file, 'lxml')
        rows = soup('row')

        for row in rows:
            # Parse post body content as HTML
            body_xml = bsoup( html.unescape( row['body'] ), 'lxml' )
            docno = row['id']
            
            # Remove unwanted <p>, <body>, <html> tags
            tagsToRemove = ['p','a','body','html']
            for tag in body_xml( tagsToRemove ):
                tag.unwrap()

            # Extract formula data
            taggedFormulas = body_xml('span')
            formulaIds = [ node['id'] for node in taggedFormulas ]

            if formula_index:
                ## Formula index entries   ##
                #  One output per formula
                for mathTag in taggedFormulas:
                    yield { 'docno':    mathTag[id],
                            'text':     mathTag.get_text(),
                            'postid':   docno
                        }
            else:
                ## Post text index entries ##
                #  Rename formula tags from 'span' to 'math,' remove unused tag attributes
                for tag in taggedFormulas:
                    tag.name = "math"
                    del tag['class']
                    del tag['id']

                # Get updated post text 
                modified_post_text = str( body_xml )

                # Note: the formula ids are stored in a string currently.
                yield { 'docno' :   docno,
                        'text' :    modified_post_text,
                        'mathids' : formulaIds
                    }
 

def show_tokens( index ):
    # Show lexicon entries
    for kv in index.getLexicon():
        print("%s :    %s" % (kv.getKey(), kv.getValue().toString()) )    
            
################################################################
# Search engine construction and search
################################################################
def search_engine( index, model, metadata_keys ):
    return pt.BatchRetrieve( index, wmodel=model, metadata = metadata_keys )

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

    return engine( query_pairs )

def test_retrieval( post_index, math_index ):
    print("[ Testing post index retrieval ]")
    posts_bm25 = search_engine( post_index, 'BM25', [ 'docno', 'text', 'mathids' ] )
    print( query( posts_bm25, 'Paul Hoffman' ) )
    print('')
    print( batch_query( posts_bm25, ['Paul Hoffman', 'proof' ] ) )
    print('')


# DEAD CODE: for reference only
# Retrieving with document text in the result table
# NOTE: Next two lines described as equivalent in documentation, 
#  but the order in output table appears not to be the same.
#bm25 = pt.BatchRetrieve(index, wmodel="BM25", metadata=["docno","text","math","mathids"])
#bm25 = pt.BatchRetrieve(index, wmodel="BM25") >> pt.text.get_text(index, 'text')
# Using search with field-based search in Terrier QL seems to be recognized -- but field changed to upper case?
#results = bm25.search('Paul Hoffman') 

################################################################
# Main function
################################################################
def main():
    # Process command line arguments
    parser = argparse.ArgumentParser(description="Indexing tool for ARQMath data.")
    parser.add_argument('-f', '--file-list', default=[], help='ARQMath XML files to index')
    parser.add_argument('-l', '--lexicon', default=False, help='show lexicon entries')
    args = parser.parse_args()

    # Start PyTerrier -- many Java classes unavailable until this is complete
    if not pt.started():
        pt.init()
    
    # Index posts and formulas in each passed file
    for file_name in args.f:
        # Post text index construction
        # Store post text and ids for formulas in each post in the 'meta' (document) index
        post_indexer = pt.IterDictIndexer( file_name,
                meta=['docno','text','mathids'],
                meta_lengths=[20, 4096, 128 ],
                overwrite=True )

        # Index only the post text for retrieval
        post_index_ref = post_indexer.index( generate_XML_post_docs( file_name ) )
        post_index = pt.IndexFactory.of( post_index_ref )

        print('Post text index statistics:')
        print( post_index.getCollectionStatistics().toString() )

        if args.l:
            show_tokens( post_index )
        print('')

        # Formula index construction
        math_indexer = pt.IterDictIndexer( file_name,
                meta=['docno','text','postid'] )
        math_index_ref = math_indexer.index( generate_XML_post_docs( file_name, formulas=True ) )
        math_index = pt.IndexFactory.of( math_index_ref )

        print('Math (formula) index statistics:')
        print( math_index.getCollectionStatistics().toString() )

        if args.l:
            show_tokens( math_index )
        print('')

    # Retrieval test
    test_retrieval( post_index, math_index )    

if __name__ == "__main__":
    main()
