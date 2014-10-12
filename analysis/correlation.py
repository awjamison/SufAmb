'''
Created on Aug 17, 2014

@author: andrew
'''

from collections import defaultdict #@UnresolvedImport
from eelbrain import Dataset, testnd, morph_source_space


def corr_sns(e, variables=['VerbGivenWord', 'Ambiguity'], ndvar=False):
    
    e.set(epoch='epoch_behav_fltr')
    corrs = defaultdict(list)
    for s in e:
        ds = e.load_epochs(baseline=(None, 0), ndvar=ndvar)
        for var in variables:
            corr = testnd.corr(ds['epochs'], ds[var], ds=ds)
            corrs[var].append(corr)
    return dict(corrs)


def corr_src(e, variables=['VerbGivenWord', 'Ambiguity'], factor='Type', output='r', ndvar=True):
    
    mri = ['R0053', 'R0606', 'R0632', 'R0684', 'R0734', 'R0845', 'R0850']
    e.set(epoch='epoch_behav_fltr', parc='')
    
    if ndvar==True: src = 'src'
    else: src = 'stc'
    
    corrs = defaultdict(list)

    if factor is None:
        for s in e:
            ds = e.load_epochs_stc(sns_baseline=(None, 0), ndvar=ndvar)
            for var in variables:
                corr = testnd.corr(ds[src], ds[var])
                stat = getattr(corr, output)
                if stat.source.subject in mri:
                    stat = morph_source_space(stat, 'fsaverage')
                else:
                    stat.source.subject = 'fsaverage'
                corrs[var].append(corr)
    else:
        for s in e:
            ds = e.load_epochs_stc(sns_baseline=(None, 0), ndvar=ndvar)
            byfac = ds.get_subsets_by(factor)
            for var in variables:
                for f in byfac:
                    corr = testnd.corr(byfac[f][src], byfac[f][var])
                    stat = getattr(corr, output)
                    if stat.source.subject in mri:
                        stat = morph_source_space(stat, 'fsaverage')
                    else:
                        stat.source.subject = 'fsaverage'
                    name = var + '_' + f
                    corrs[name].append(stat) # level of factor
                corr = testnd.corr(ds[src], ds[var])
                stat = getattr(corr, output)
                if stat.source.subject in mri:
                    stat = morph_source_space(stat, 'fsaverage')
                else:
                    stat.source.subject = 'fsaverage'
                corrs[var].append(corr) # collapsed across levels
    
    # turn defaultdict into dataset
    corrs = dict(corrs)
    combined = Dataset()            
    for k in corrs:
        combined[k] = corrs[k]
        
    return combined





##################### recipes #####################

### t test on correlations in source space
# corrs = corr_src(e, ndvar=True)
# r = corr2stat(corrs)
# rAmb = Dataset(r['Ambiguity'])
# res = testnd.ttest_1samp(Y=rAmb['src corr Ambiguity'])
# eel.plot.TopoButterfly(res.diff_p_uncorrected)
# eel.plot.brain.stat(res.diff_p_uncorrected, p1=0.01)


### pairwise correlation differences within a factor
# ds = eel.load.unpickle('/Volumes/BackUp/sufAmb/pickled/r_ds.pickled')
# Y = ds['VerbGivenWord']
# Y.source.set_parc('frontal_temporal')
# index = Y.source.parc.isnotin(['unknown-lh', 'unknown-rh'])
# Y = Y.sub(source=index)
# testnd.ttest_1samp(Y, match='subject', ds=ds, samples=100, parc='source', tstart=0.13, pmin=0.05)