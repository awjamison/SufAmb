'''
Created on Aug 19, 2014

@author: andrew
'''

import matplotlib
matplotlib.use('wxagg')
from eelbrain import load, plot
from load_sig_clusters import sig_clusters

_dirc = '/Volumes/BackUp/sufAmb/pickled/'

#sigv = sig_clusters(pmin=0.05)
#sigc = sigv['ols_VerbGivenWord_main'][0]['cluster']

sigc = load.unpickle('/Volumes/BackUp/sufAmb/corr_cluster_timespace/corr_Ambiguity_stemVstemEd.pickled')
sigc = sigc.clusters[0]['cluster']

extent = sigc.sum('time')

def plot_extent():
    p = plot.brain.cluster(extent, surf='white')
    return p

index = extent != 0
path = _dirc+'ols_VerbGivenWord.pickled'
betas = load.unpickle(path)
timecourse = betas['beta'].sum(index)

def plot_tc():
    p = plot.UTSStat(timecourse, match='subject', ds=betas.sub("Type=='all'"))
    return p
    
def plot_tc_Type():
    p = plot.UTSStat(timecourse, X='Type', match='subject', ds=betas.sub("Type!='all'"))
    return p
