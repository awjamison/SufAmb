'''
Created on Aug 19, 2014

@author: andrew
WARNING: needs work to extend to general case of IVs and regions
'''

from eelbrain import save, load, testnd
from load_sig_clusters import sig_clusters

dr = '/Volumes/Backup/sufAmb/pickled/'
IVs = ['VerbGivenWord']
regions = ['ols_VerbGivenWord_main']
test = load.unpickle(dr+regions[0]+'.pickled')
test = test.clusters[0]


match='subject'
samples=10000
pmin=0.15
mintime=0.03
X = 'Type'

tests = {}
for r, v in zip(regions, IVs):
    print v
    testsv = {}
    sigc = test['cluster']
    tstart = test['tstart']
    tstop = test['tstop']
    extent = sigc.sum('time')
    index = extent != 0
    
    fil = dr+'ols_'+v+'.pickled'
    data = load.unpickle(fil)
    Y = data['beta'].sum(index) # betas collapsed across src
    
    print 'test 1'
    testsv['stemVstemS_time'] = testnd.ttest_rel(Y=Y, ds=data, X=X, c1='stem', c0='stemS', match=match, samples=samples, pmin=pmin, tstart=tstart, tstop=tstop, mintime=mintime)
    print 'test 2'
    testsv['stemVstemEd_time'] = testnd.ttest_rel(Y=Y, ds=data, X=X, c1='stem', c0='stemEd', match=match, samples=samples, pmin=pmin, tstart=tstart, tstop=tstop, mintime=mintime)
    print 'test 3'
    testsv['stemSVstemEd_time'] = testnd.ttest_rel(Y=Y, ds=data, X=X, c1='stemS', c0='stemEd', match=match, samples=samples, pmin=pmin, tstart=tstart, tstop=tstop, mintime=mintime)
    for t in testsv:
        name = v+'_'+t
        path = '/Volumes/BackUp/sufAmb/cluster_time/ols_%s.pickled' % name
        save.pickle(testsv[t], path)
    
    tests[v] = testsv