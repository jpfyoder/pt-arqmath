################################################################
# math_recoding.py
#
# Dictionaries and functions to recode formula encodings (e.g.,
# LaTeX) for use with PyTerrier, which removes punctuation by
# default.
#
# Author:
# R. Zanibbi, April 2022 (CSCI 539, Information Retrieval, RIT)
################################################################

################################################################
# LaTeX Symbol Map
################################################################
# Warning - mapping may be incomplete.
# Mapping ignores backslash (\) in TeX commands
# Commands will be tokenized as-is with no backslash 
# (e.g., \geq -> ge )
latex_symbol_map = {
        "+" : " mplus ",
        "-" : " mminus ",
        "^" : " mpower ",
        "*" : " mstar ",

        "#" : " mhash ",
        "!" : " mexclaim ",
        "?" : " mquestion ", 
        "," : " mcomma ",
        "." : " mperiod ",
        "/" : " mfslash ",
        ":" : " mcolon ",
        ";" : " msemicolon ",
        "&" : " mampersand ",
        "~" : " mtilde ",
        "`" : " mbackquote ",
        "@" : " mat ",
        "$" : " mdollar ",
        "%" : " mpercent ",
        '"' : " mdquote ",
        "'" : " msquote ",
        "|" : " mvbar ",

        "=" : " mequals ",
        ">" : " mgreater ",
        "<" : " mlesser ",

        "{" : " mobrace ",
        "}" : " mcbrace ",
        "[" : " mosquareb ",
        "]" : " mcsquareb ",
        "(" : " moparen ",
        ")" : " mcparen ",
        
        "\\\\" : " mdblbackslash ",
        "\\" : " "
    }

# 'Mini' query language for formulas, replacing pyterrier ops        
# e.g., using '_ptand' for the AND (+) operator
# e.g., using '_ptnot' for the NOT (-) operator
pyterrier_symbol_map = {
        "_pand":       "+",
        "_pnot":       "-",
        "_pobrace":    "{",
        "_pcbrace":    "}"
    }


################################################################
# Functions to transform strings encoding math
################################################################
def rewrite_symbols( in_string, map_dict ):
    out_string = in_string

    # Brute force - apply rules one-at-a-time
    for key in map_dict:
        out_string = out_string.replace( key, map_dict[ key ] )

    return out_string

def translate_latex( TeXstring ):
    # Replace LaTeX symbols in a string by text tokens
    return rewrite_symbols( TeXstring, latex_symbol_map )

def translate_query( query):
    # Translate query string to 'arqmath pyterrier query language' representation
    # (translate TeX symbols + 'meta' operators (e.g., _pand for '+' in pyterrier query language)
    return rewrite_symbols(
            rewrite_symbols( query, latex_symbol_map ),
            pyterrier_symbol_map 
        )

def translate_qlist( query_list ):
    # For batch retrieval
    return list( map( translate_query, query_list ) )
