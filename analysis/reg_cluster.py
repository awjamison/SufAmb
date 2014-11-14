'''
Created on Aug 18, 2014

@author: andrew
'''

import eelbrain
from eelbrain import save, load, testnd

def reg_clusters_t_src(mod, IV, factor=None, c1=None, c0=None):
    """ Cluster permutation (src x time) t-tests on regression coefficients.
    Example: mod='main-VerbGivenWord', IV='LogFre'
    """

    dr = '/Volumes/Backup/sufAmb/regression/'
    f = dr+'ols_'+mod+'.pickled'
    data = load.unpickle(f)

    # cluster permutation parameters
    Y='beta'
    match='subject'
    samples=100
    pmin=0.05
    tstart=0.13
    tstop=0.45
    mintime=0.03
    minsource=10

    if factor is None:
        test = testnd.ttest_1samp(Y=Y, ds=data, match=match, samples=samples, pmin=pmin,
                                  tstart=tstart, tstop=tstop, mintime=mintime, minsource=minsource)
    elif factor == 'main':
        test = testnd.ttest_1samp(Y=Y, ds=data.sub("condition=='main'"), match=match, samples=samples, pmin=pmin,
                                  tstart=tstart, tstop=tstop, mintime=mintime, minsource=minsource)
    else:
        test = testnd.ttest_rel(Y=Y, ds=data, X=factor, c1=c1, c0=c0, match=match, samples=samples, pmin=pmin,
                                tstart=tstart, tstop=tstop,  mintime=mintime, minsource=minsource)

    print "Finished cluster test: mod=%s, IV=%s, factor=%s, c1=%s, c0=%s" % (mod, IV, factor, c1, c0)
    path = "/Volumes/BackUp/sufAmb/reg-cluster_time_src/ols_mod-%s_IV-%s_factor-%s_c1-%s_c0-%s.pickled" \
        % (mod, IV, factor, c1, c0)
    save.pickle(test, path)

    return test
