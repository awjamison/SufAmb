'''
Created on Aug 17, 2014

@author: andrew
'''

from eelbrain import Dataset, Factor, morph_source_space, combine

def ols(e, IVs, DV='src', fac='Type', ndvar=True, parc='frontal_temporal'):
    """Currently this only handles a single factor
    """
    
    e.set(epoch='epoch_behav_fltr', parc=parc)
    
    mri = ['R0053', 'R0606', 'R0632', 'R0684', 'R0734', 'R0845', 'R0850']
    
    # try to catch errors in input arguments early
    if DV != 'src':
        raise NotImplementedError('untested for DV != src.')
    if not ndvar:
        raise NotImplementedError('untested for ndvar != True.')
    if not hasattr(IVs, '__getitem__'):
        raise TypeError('IVs is of type %s, which has no attribute __getitem__.' % type(IVs))
    if not hasattr(IVs, '__iter__'):
        raise TypeError('IVs is of type %s, which has no attribute __iter__.' % type(IVs))
    events = e.load_events()
    for iv in IVs:
        if events.has_key(iv) is False:
            raise ValueError('%s is not in the log file.' % iv)
    del events
    for s in e:
        if s in mri: continue
        e.load_annot()  # raises an error if parc file not found in fsaverage
        break

    # WARNING NOTE: row values are coupled only by the for loop; may use dict in future
    betas = []
    vars = []
    factors = []
    subjects = []

    for s in e:
        print s
        
        # avoid IOError if parc only exists in fsaverage; parc will be reset below after loading epochs
        e.set(parc='')
        
        # load src epochs
        ds = e.load_epochs_stc(sns_baseline=(None, 0), ndvar=ndvar)
        
        # create smaller dataset that is just the parcellation
        if s in mri:
            ds[DV] = morph_source_space(ds[DV], 'fsaverage')
        else:
            ds[DV].source.subject = 'fsaverage'
        ds[DV].source.set_parc(parc)  # reset parc
        index = ds[DV].source.parc.isnotin(['unknown-lh', 'unknown-rh'])  # remove these
        ds[DV] = ds[DV].sub(source=index)
        
        # regress main effect
        mod = tuple(ds[iv] for iv in IVs)
        b = ds[DV].ols(mod)  # fit multiple ols to predictor vars
        for name, value in zip(IVs, b):
            vars.append(name)
            betas.append(value)
            factors.append('all')
            subjects.append(s)
        
        # regress at each level of fac
        if fac is not None:
            levels = ds[fac].cells
            for l in levels:
                index = "%s == '%s'" % (fac, l)
                lev = ds.sub(index)
                mod = tuple(lev[iv] for iv in IVs)
                b = lev[DV].ols(mod)
                for name, value in zip(IVs, b):
                    vars.append(name)
                    betas.append(value)
                    factors.append(l)
                    subjects.append(s)
        else:
            levels = ()
        # after each iteration of e, number of rows in betas_ds is increased by (n_levels+1) * n_vars
        
    betas = combine(betas)  # combine list of NDVars into single NDVar
    vars = Factor(vars, name='predictors', random=False)
    factors = Factor(factors, name='levels', random=False)
    subjects = Factor(subjects, name='subjects', random=True)
    betas_ds = Dataset(('subject', subjects), (fac, factors), ('var', vars), ('beta', betas),
                       info={'predictors': IVs, 'outcome': DV, 'factor': fac})
        
    return betas_ds
