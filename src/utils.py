################################################################
# utils.py
#
# Helper functions
#
# Author:
# R. Zanibbi, Apr 2022
################################################################

# Functions for debugging
def pshow( message, var ):
    dbshow( message, var, pause=True )

def dbshow( message, var, pause=False):
    print( message + ":")
    print( str(var))

    if pause == True:
        input("Press any key to continue...")
