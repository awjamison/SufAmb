'''
Created on Aug 17, 2014

@author: andrew
'''

from collections import defaultdict #@UnresolvedImport
from eelbrain import Dataset, Factor, morph_source_space, combine
from numpy.linalg.linalg import LinAlgError #@UnresolvedImport


def __ols(e, IV, DV='src', covar=[], fac='Type', ndvar=True, parc='frontal_temporal'):
    
    e.set(epoch='epoch_behav_fltr', parc=parc)
    
    mri = ['R0053', 'R0606', 'R0632', 'R0684', 'R0734', 'R0845', 'R0850']
    
    # check if input parameters are ok
    if DV != 'src':
        print "WARNING: untested for DV != 'src'"
    if not hasattr(covar, '__getitem__'):
        raise TypeError('covar is of type %s, which has no attribute __getitem__' % type(covar))
    if not hasattr(covar, '__iter__'):
        raise TypeError('covar is of type %s, which has no attribute __iter__' % type(covar))     
    if not ndvar:
        raise ValueError('ndvar must be True')
    for s in e:
        if s in mri: continue
        e.load_annot() # raises an error if parcellation file not found in fsaverage
        break
    
    betas = defaultdict(list)
    subjects = Factor([s for s in e], random=True)
    betas_ds = Dataset( ('subject', subjects), info={'predictor': IV, 'outcome': DV, 'covariates': covar, 'factor':fac} )

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
        
        # regress main effect
        if len(covar) > 0:
            # regress DV on covariate
            cov = [ds[covar[i]] for i in range(len(covar))]
            resid = ds[DV].residuals(cov)
            # regress IV on residuals from previous regression
            beta = resid.ols(ds[IV])
        else:
            beta = ds[DV].ols(ds[IV])
        betas['all'].append(beta)
        
        # regress at each level of fac
        if fac is not None:
            levels = ds[fac].cells
            for l in levels:
                index = "%s == '%s'" % (fac, l)
                lev = ds.sub(index)
                if len(covar) > 0:
                    # regress DV on covariate
                    cov = [lev[covar[i]] for i in range(len(covar))]
                    resid = lev[DV].residuals(cov)
                    # regress IV on residualevs from previous regression
                    beta = resid.ols(lev[IV])
                else:
                    beta = lev[DV].ols(lev[IV])
                betas[l].append(beta)
        
    for x in betas:
        betas_ds[x] = combine(betas[x])
        
    return betas_ds




def ols(e, IV, DV='src', covar=[], fac='Type', ndvar=True, parc='frontal_temporal'):
    """Currently this only handles a single factor
    """
    
    e.set(epoch='epoch_behav_fltr', parc=parc)
    
    mri = ['R0053', 'R0606', 'R0632', 'R0684', 'R0734', 'R0845', 'R0850']
    
    # try to catch errors in input arguments early
    if DV != 'src':
        print "WARNING: untested for DV != 'src'"
    if not hasattr(covar, '__getitem__'):
        raise TypeError('covar is of type %s, which has no attribute __getitem__' % type(covar))
    if not hasattr(covar, '__iter__'):
        raise TypeError('covar is of type %s, which has no attribute __iter__' % type(covar))     
    if not ndvar:
        raise ValueError('ndvar must be True')
    for s in e:
        if s in mri: continue
        e.load_annot() # raises an error if parcellation file not found in fsaverage
        break
    
    # WARNING NOTE: row values are coupled only by the for loop; may use dict in future
    betas = []
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
        
        # regress main effect
        if len(covar) > 0:
            try:
                # regress DV on covariate
                cov = [ds[covar[i]] for i in range(len(covar))]
                resid = ds[DV].residuals(cov)
                # regress IV on residuals from previous regression
                beta = resid.ols(ds[IV])
            except LinAlgError:
                raise LinAlgError
                print "Singular matrix; two or more covariates are collinear."
                
        else:
            beta = ds[DV].ols(ds[IV])
        factor.append('all')
        betas.append(beta)
        
        # regress at each level of fac
        if fac is not None:
            levels = ds[fac].cells
            for l in levels:
                index = "%s == '%s'" % (fac, l)
                lev = ds.sub(index)
                if len(covar) > 0:
                    try:
                        # regress DV on covariate
                        cov = [lev[covar[i]] for i in range(len(covar))]
                        resid = lev[DV].residuals(cov)
                        # regress IV on residualevs from previous regression
                        beta = resid.ols(lev[IV])
                    except LinAlgError:
                        raise LinAlgError
                        print "Singular matrix; two or more covariates are collinear."
                else:
                    beta = lev[DV].ols(lev[IV])
                factor.append(l)
                betas.append(beta)
        
        facLen = len(levels)+1 # add 1 for main effect (collapsed across levels)
        subjects.extend( [s]*facLen )
        # after each iteration of s, number of rows in betas_ds is increased by number of levels in factor
        
    betas = combine(betas)
    factor = Factor(factor, name=fac, random=False)
    subjects = Factor(subjects, name='subjects', random=True)
    betas_ds = Dataset( ('subjects',subjects), (fac, factor), ('betas',betas), info={'predictor': IV, 'outcome': DV, 'covariates': covar, 'factor':fac} )
        
    return betas_ds


    