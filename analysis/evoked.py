from eelbrain import combine

def evoked_sns(e):

    e.set(epoch='epoch_behav_fltr', model='Type%DominanceCategory')
    dss = []
    for s in e:
        ds = e.load_evoked(baseline=(None, 0)) # baseline correct
        dss.append(ds)
         
    return combine(dss)

def evoked_stc(e):

    e.set(epoch='epoch_behav_fltr', model='Type%DominanceCategory')
    dss = []
    for s in e:
        ds = e.load_evoked_stc(sns_baseline=(None, 0), morph_ndvar=True)
        dss.append(ds)
         
    return combine(dss)





##################### recipes #####################

# Butterfly plot of sensor space collapsed over all conditions
# eel.plot.Butterfly(sen['meg'])

# Whole brain plot of source space collapsed over all conditions
# eel.plot.brain.activation(src['srcm'])

# Src plot of p-vals for paired t-test
# res = testnd.ttest_rel(Y='srcm', X='Type', c1='stem', c0='stemS', match='subject', ds=src)
# eel.plot.brain.stat(res.p_uncorrected, p1=0.01)

# TopoMap for ANOVA at t=M170 peak for each subject
# res = testnd.anova('m170_time', 'Type*Subject', match='Subject', ds=dss)
# eel.plot.Topomap(res.p_uncorrected)
