'''
Created on Aug 21, 2014

@author: andrew
'''

from eelbrain import Dataset, Factor, morph_source_space, combine, testnd


def ols(e, IV, DV='src', covar=None, fac='Type', ndvar=True, parc='frontal_temporal'):
    """COV IS CURRENTLY NOT SUPPORTED!!!
    """
    
    if covar:
        raise NotImplementedError('Covariates are currently not supported')
    
    e.set(epoch='epoch_behav_fltr', parc=parc)
    mri = ['R0053', 'R0606', 'R0632', 'R0684', 'R0734', 'R0845', 'R0850']
    
    # try to catch errors in input arguments early
    if DV != 'src':
        print "WARNING: untested for DV != 'src'"
    if not ndvar:
        raise ValueError('ndvar must be True')
    for s in e:
        if s in mri: continue
        e.load_annot() # raises an error if parcellation file not found in fsaverage
        break
    
    # WARNING NOTE: row values are coupled only by the for loop; may use dict in future
    corrs = []
    factor = []
    subjects = []

    for s in e:
        print s
        
        # avoid IOError if parc only exists in fsaverage; parc will be reset below after loading epochs
        e.set(parc='')
        
        # load src epochs
        ds = e.load_epochs_stc(sns_baseline=(None, 0), ndvar=ndvar)
        
        # create smaller dataset from parcellation
        if s in mri:
            ds[DV] = morph_source_space(ds[DV], 'fsaverage')
        else:
            ds[DV].source.subject = 'fsaverage'
        ds[DV].source.set_parc(parc) # reset parc
        index = ds[DV].source.parc.isnotin(['unknown-lh', 'unknown-rh'])
        ds[DV] = ds[DV].sub(source=index)
        
        corr = testnd.corr(ds[DV], ds[IV])
        corr = getattr(corr, 'r')
        factor.append('all')
        corrs.append(corr)
        
        # regress at each level of fac
        if fac is not None:
            levels = ds[fac].cells
            for l in levels:
                index = "%s == '%s'" % (fac, l)
                lev = ds.sub(index)

                corr = testnd.corr(lev[DV], lev[IV])
                corr = getattr(corr, 'r')
                factor.append(l)
                corrs.append(corr)
        
        facLen = len(levels)+1 # add 1 for main effect (collapsed across levels)
        subjects.extend( [s]*facLen )
        # after each iteration of s, number of rows in corrs_ds is increased by number of levels in factor
        
    corrs = combine(corrs)
    factor = Factor(factor, name=fac, random=False)
    subjects = Factor(subjects, name='subjects', random=True)
    corrs_ds = Dataset( ('subjects',subjects), (fac, factor), ('corrs',corrs), info={'predictor': IV, 'outcome': DV, 'covariates': covar, 'factor':fac} )
        
    return corrs_ds
