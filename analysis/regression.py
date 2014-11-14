'''
Created on Aug 17, 2014

@author: andrew
'''

from eelbrain import Dataset, Factor, morph_source_space, combine

def ols(e, IVs, DV='src', factor='Type', level='all', ndvar=True, parc='frontal_temporal'):
    """OLS for main effect, level of a factor, or main effect and all levels of a factor.\n
    \n
    Fits a separate model to each condition (level), so values of a particular variable may not be comparable across\n
    conditions, i.e. you can't do an ANCOVA this way. Instead, dummy code the conditions in the model matrix and\n
    compare these within the condition 'main'.
    Notes: currently this only handles a single factor.
    """
    
    e.set(epoch='epoch_behav_fltr', parc=parc)
    
    mri = ['R0053', 'R0606', 'R0632', 'R0684', 'R0734', 'R0845', 'R0850']
    
    # try to catch errors in input arguments early
    if DV != "src":
        raise NotImplementedError("untested for DV != src.")
    if not ndvar:
        raise NotImplementedError("untested for ndvar != True.")
    if not hasattr(IVs, "__getitem__"):
        raise TypeError("IVs is of type %s, which has no attribute __getitem__." % type(IVs))
    if not hasattr(IVs, "__iter__"):
        raise TypeError("IVs is of type %s, which has no attribute __iter__." % type(IVs))
    events = e.load_events()
    for iv in IVs:
        if events.has_key(iv) is False:
            raise ValueError("Variable %s is not in the log file." % iv)
    if factor is not None:
        if events.has_key(factor) is False:
            raise ValueError("Factor %s is not in the log file." % factor)
        if not level in events[factor].cells or level == 'all':
            raise ValueError("Level is not recognized.")
    else:
        if not level == "all":
            raise ValueError("Level must be == 'all' when factor == None.")
    del events
    for s in e:
        if s in mri: continue
        e.load_annot()  # raises an error if parc file not found in fsaverage
        break

    # each of these lists is a column; number of rows == len(betas)
    betas = []
    vars = []
    conditions = []
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
        
        # set which levels are to be used
        if factor is None:
            levels = ['main']
        else:
            if level == 'all':
                levels == ['main'] + (list(ds[factor].cells))
            else:
                levels = [level]
        
        # regress IVs within each level
        for l in levels:
            if l == 'main':
                lev = ds
            else:
                index = "%s == '%s'" % (factor, l)
                lev = ds.sub(index)
            mod = tuple(lev[iv] for iv in IVs)
            b = lev[DV].ols(mod)
            for name, value in zip(IVs, b):
                vars.append(name)
                betas.append(value)
                conditions.append(l)
                subjects.append(s)
        # after each iteration of e, number of rows in betas_ds is increased by n_levels * n_vars
        
    betas = combine(betas)  # combine list of NDVars into single NDVar
    vars = Factor(vars, name='predictors', random=False)
    conditions = Factor(conditions, name='levels', random=False)
    subjects = Factor(subjects, name='subjects', random=True)
    betas_ds = Dataset(('subject', subjects), ('condition', conditions), ('predictor', vars), ('beta', betas),
                       info={'predictors': IVs, 'outcome': DV, 'factor': factor})
        
    return betas_ds
