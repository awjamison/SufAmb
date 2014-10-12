from collections import defaultdict
import os
import numpy as np
from eelbrain.lab import load, save, Var, Factor

masterPath = os.path.join('/Volumes','BackUp','sufAmb_stimuli','stimuli',
                          'stim_properties','master_list35.txt')

masterRT = load.tsv(masterPath) # complete set of stimuli and properties
masterRT = masterRT[masterRT['Word'].isnot('spread', 'spreads')] # get rid of these @!$*ers !!!

# POS entropy for stem, word, lemma
def stm_pos(ds):
    """
    Definition: H(@) = -p(N|@)*log2(p(N|@))-p(V|@)*log2(p(V|@))
    Add 1 for smoothing
    """
    
    pN = ds['fre.S|@'].x
    pV = ds['fre.i|@'].x
    H = -pN * np.log2(pN+1) -pV * np.log2(pV+1)
    
    return Var(H)

def wrd_pos(ds):
    """
    Definition: H(@+x) = -p(N|@+x)*log2(p(N|@+x))-p(V|@+x)*log2(p(V|@+x))
    Add 1 for smoothing
    """
    
    pN = ds['fre.P|@+s'].x
    pV = ds['fre.e3S|@+s'].x
    H = -pN * np.log2(pN+1) -pV * np.log2(pV+1)
    
    return Var(H)

def lem_pos(ds):
    """
    Definition: H(@*) = -p(N|@*)*log2(p(N|@*))-p(V|@*)*log2(p(V|@*))
    Add 1 for smoothing
    """
    
    pN = ds['fre.1|@*'].x
    pV = ds['fre.4|@*'].x
    H = -pN * np.log2(pN+1) -pV * np.log2(pV+1)
    
    return Var(H)

def add_vars(ds=masterRT):
    
    StmPOSEntropy = stm_pos(ds)
    WrdPOSEntropy = wrd_pos(ds)
    LemPOSEntropy = lem_pos(ds)
    
    ds['StmPOSEntropy'] = StmPOSEntropy
    ds['WrdPOSEntropy'] = WrdPOSEntropy
    ds['LemPOSEntropy'] = LemPOSEntropy
    
    ds.save_txt(masterPath)
    
    return ds