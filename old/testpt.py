import pyterrier as pt
import pandas as pd
import os

##################################
# Configuration
##################################
# Avoid truncating output too much
pd.set_option('display.max_colwidth', 150)

##################################
# Data Definitions
##################################

# From ECIR 2021 tutorial
# Appears to be lists followed by index position (column) names
docs_df = pd.DataFrame([
        ["d1", "this is the first document of many documents"],
        ["d2", "this is another document"],
        ["d3", "the topic of this document is unknown"]
    ], columns=["docno", "text"])


##################################
# Main Function
##################################
def main():
    # RZ: Important note: until PyTerrier is initialized, some attributes
    # do not exist; guessing these are linked with Terrier (Java system)
    if not pt.started():
        pt.init()

    # WARNING: directory name is automatically converted to lower case!
    indexer = pt.DFIndexer("./test-index", overwrite=True)
    index_ref = indexer.index(docs_df["text"], docs_df["docno"])
    print( index_ref.toString())

    # Load index (Terrier Java Index object -- see Javadocs)
    index = pt.IndexFactory.of(index_ref)
    print(index.getCollectionStatistics().toString())

    # Show lexicon entries
    for kv in index.getLexicon():
        #print("%s (%s) ->\n  %s\n  (%s)" % 
        #        (kv.getKey(), 
        #        type(kv.getKey()), 
        #        kv.getValue().toString(), 
        #        type(kv.getValue()) ) )
        print("%s ->  %s" %
                (kv.getKey(), 
                kv.getValue().toString()) ) 

    br = pt.BatchRetrieve(index, wmodel="BM25")
    results = br.search("document")
    print( results )

    queries  = pd.DataFrame( [
        [ "q1", "document"],
        [ "q2", "first document"]],
        columns=[ "qid", "query" ] )

    # 'transform' method called by default, arg 'queries' to that fn here
    queries = br(queries)
    print (queries)


if __name__== "__main__":
    print("Running...")
    main()

