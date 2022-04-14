################################################################
# arqmath_topics_rels.py
#
# Functions to generate Pandas dataframes for ARQMath topics 
# and qrel (Query Relevance) files for use with PyTerrier.
#
# Author:
# R. Zanibbi, Apr 2022
################################################################
import pyterrier.io
import os
from bs4 import BeautifulSoup as bsoup
import sys
import html
import pandas as pd
from index_arqmath import *

def load_qrels( file_name ):
    return pyterrier.io.read_qrels( file_name )

# HACK: modifying code from index file to work with topics, no outputs
# Removes ids (which have no correspondence in index)
# WARNING: Assumes well-formed topic entries
def replace_formulas( soup_tag ):
    # Skip span tags without id's (i.e., formulas without identifiers)
    formulaTags = soup_tag('span')
    for tag in formulaTags:
        tag.name = 'math'
        del tag['class']
        del tag['id']


def convert_topic( topic_tag ):
    # Topic number and tags
    topic_number = topic_tag['number']
    tags = ''
    if topic_tag.tags:
        tags = html.unescape( topic_tag.tags.get_text() )   

    # Convert fields to index representation
    # HACK: Removing dollar signs for formulas (appear absent in the index)
    # DEBUG: Beautiful soup 'get_text()' removes all tags; use str for full trees
    title_text =  html.unescape( str( topic_tag('title')[0] )).replace('$','')
    body_text =   html.unescape( str( topic_tag('question')[0])).replace('$','')

    title_soup = bsoup( title_text,  'lxml')
    body_soup = bsoup( body_text, 'lxml')

    replace_formulas( title_soup )
    replace_formulas( body_soup )

    # Remove pruned tags (defined in index_arqmath.py)
    remove_tags( title_soup, TAGS_TO_REMOVE )
    remove_tags( body_soup, TAGS_TO_REMOVE )

    # Save full query as all converted topic content (all fields)
    # REWRITE punctuation/math before exporting
    # NOTE: To see original query, please consult original topics files
    title_text = rewrite_symbols( title_soup.get_text(), latex_symbol_map )
    body_text =  rewrite_symbols(body_soup.get_text(), latex_symbol_map )
    topic_text = 'Title: ' + title_text + \
            ' Body: ' + body_text + \
            ' Tags: ' + tags

    # BUG: the output text does not contain the 'math' tokens for math tags present
    # in the indexed documents. Skipping for time (BM25 is a bag-of-words model).
    return ( topic_number, topic_text, title_text, body_text, tags )

def read_topic_file( file_name ):
    with open( file_name ) as infile:
        topic_soup = bsoup( infile, 'lxml' )
        topic_tags = topic_soup('topic')

        # Lists to hold output data
        tnum_list = []
        ttext_list = []
        title_list = []
        question_list = []

        # Extract fields
        tags_list = []
        for tag in topic_tags:
            ( tnum, ttext, title, question, tags ) = convert_topic( tag )
            tnum_list.append( tnum )
            ttext_list.append(ttext)
            title_list.append( title )
            question_list.append( question )
            tags_list.append( tags )

        df = pd.DataFrame( list( zip( tnum_list, ttext_list, title_list, question_list, tags_list ) ),
                columns = [ 'qid','query','title','body','tags' ] )

    return ( len( topic_tags ), df )

     
def main():
    # Process arguments
    inTopics = sys.argv[1]
    inQrels = sys.argv[2]

    # Set display width for Pandas
    pd.set_option( 'display.max.colwidth', 150 )
    ( num_topics, topic_df ) = read_topic_file( inTopics )

    print("Topic File: " + inTopics + "   Number of Topics: " + str(num_topics) )
    print("\n>>> TAGS")
    print(topic_df['tags'])

    print("\n>>> TITLES")
    print(topic_df['title'])

    print("\n>>> QUESTION BODIES")
    print(topic_df['body'])

    qrels = load_qrels( inQrels )
    print("\n>>> QREL Contents:")
    print( qrels )

if __name__ == "__main__":
    main()
