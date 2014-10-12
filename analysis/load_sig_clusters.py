'''
Created on Aug 20, 2014

@author: andrew
'''

import os
from eelbrain import load

_dirc = '/Volumes/BackUp/sufAmb/corr_cluster_timespace/'

def sig_clusters(dirc=_dirc, prefix='corr', suffix='', pmin=0.05):

    pikld = os.listdir(dirc)
    pikld = [p for p in pikld if p.startswith(prefix)==True]
    pikld = [p for p in pikld if p.endswith(suffix)==True]
    pikld = [p.replace('.pickled', '') for p in pikld]
    data = {p:load.unpickle(dirc+p+'.pickled') for p in pikld}
    data = {d:data[d] for d in data if hasattr(data[d], 'clusters')==True} # only keep results of cluster tests
    try:
        sig = {d:data[d] for d in data if data[d].clusters['p'].min() <= pmin} # ds with a/l one sig cluster
        sig = {d:[ data[d].clusters[c] ] for d in data for c in range(data[d].clusters.n_cases) if data[d].clusters[c]['p'] <= pmin } # retrieve sig clusters from tests
        # creates this structure: {v1: [c1, c2, ...], v2: [c1, c2, ...], ...}
        # where v is a regressor and c is a cluster NDVar (with info included)
    except ValueError: # no sig clusters
        print 'No significant clusters.'
        sig = {}
    return sig
