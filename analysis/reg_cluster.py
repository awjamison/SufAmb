'''
Created on Aug 18, 2014

@author: andrew
'''

from eelbrain import save, load, testnd

#eelbrain._set_log_level('debug')

dr = '/Volumes/Backup/sufAmb/pickled/'
IV_alls = ['VerbGivenWord_nocov', 'VerbGivenWord', 'Ambiguity', 'WrdVerbyWeighted', 'WrdBiasWeighted']
IVs = ['VerbGivenWord', 'Ambiguity']

Y='beta'
match='subject'
samples=1000
pmin=0.04
tstart=0.13
tstop=0.45
mintime=0.03
minsource=10
X = 'Type'

tests = {}
for v in IVs:
    print v
    testsv = {}
    fil = dr+'ols_'+v+'.pickled'
    data = load.unpickle(fil)
    
    print 'main'
    testsv['main'] = testnd.ttest_1samp(Y=Y, ds=data.sub("Type=='all'"), match=match, samples=samples, pmin=pmin, tstart=tstart, tstop=tstop, mintime=mintime, minsource=minsource)
    #===========================================================================
    # print 'test 1'
    # testsv['stemVstemS'] = testnd.ttest_rel(Y=Y, ds=data, X=X, c1='stem', c0='stemS', match=match, samples=samples, pmin=pmin, tstart=tstart, tstop=tstop,  mintime=mintime, minsource=minsource)
    # print 'test 2'
    # testsv['stemVstemEd'] = testnd.ttest_rel(Y=Y, ds=data, X=X, c1='stem', c0='stemEd', match=match, samples=samples, pmin=pmin, tstart=tstart, tstop=tstop, mintime=mintime, minsource=minsource)
    # print 'test 3'
    # testsv['stemSVstemEd'] = testnd.ttest_rel(Y=Y, ds=data, X=X, c1='stemS', c0='stemEd', match=match, samples=samples, pmin=pmin, tstart=tstart, tstop=tstop, mintime=mintime, minsource=minsource)
    # 
    #===========================================================================
    for t in testsv:
        name = v+'_'+t
        path = '/Volumes/BackUp/sufAmb/pickled/ols_%s.pickled' % name
        save.pickle(testsv[t], path)
    
    tests[v] = testsv

