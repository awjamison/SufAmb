
import os
import socket
from mne import set_log_level
from numpy import logical_and #@UnresolvedImport

import matplotlib
matplotlib.use('wxagg')

from eelbrain import Factor, Var, load
from eelbrain.experiment import MneExperiment

##### MAKE FIFF FILE #####
# mne.gui.kit2fiff()

##### COREGISTRATION #####
# mne.gui.coregistration(), batch: mne coreg

bad_channels = ['MEG 019', 'MEG 041', 'MEG 065', 'MEG 153']

class SufAmb(MneExperiment):
    '''
    Class for Andrew's sufAmb experiment.
    '''

    groups = {'test':['R0053', 'R0606']}
    
    # R0632: weird sensor space grand average
    # R0689: excessive noise
    # R0851: excessive noise, metal in clothing
    exclude = {'subject':['R0632', 'R0689', 'R0851']}
    
    bad_channels = bad_channels
    
    epochs = {
              'epoch': dict(sel="BeforeOrAfter.isnot('0','128')", tmin=-0.15, tmax=0.6),
              'epochbl': dict(sel_epoch='epoch', tmin=-0.15, tmax=0),
              'epoch450': dict(sel="BeforeOrAfter != '128'", tmin=-0.15, tmax=0.45),
              'epoch450bl': dict(sel_epoch='epoch450', tmin=-0.15, tmax=0),
              'epoch_behav_fltr': dict(sel_epoch='epoch450', sel="GoodResponse == 1", tmin=-0.15, tmax=0.45),
              }

    _eog_sns = ['MEG 143', 'MEG 151']
    
    _values = {'experiment': ('sufAmb'),

               'log-dir': os.path.join('{meg-dir}', 'log'),
               'log-file': os.path.join('{log-dir}', 
                                        '{subject}_s*_stimuli_log.txt'),
               }
               
    _defaults = {'cov': 'epoch450bl',
                 'raw': '1-40',
                 'epoch': 'epoch_behav_fltr',
                 'rej': 'man',
                 'model': 'Condition',
                 'parc': 'frontal_temporal',
                 'src': 'ico-4'
                 }
    
    tests = {'TypeXDominanceCategory':('anova', 'Type%DominanceCategory', 'Type*DominanceCategory*subject'),
             }
    
    parcs = ['', 'frontal_temporal']

    def label_events(self, ds, experiment, subject):
        """
        Returns
        -------
        ds : Dataset
            Same object as input ds, modified in place.
        """
        ds = MneExperiment.label_events(self, ds, experiment, subject)
        
        dontinclude = ds['trigger'].isnot(128)
        ds = ds[dontinclude]
        
        log = self.load_log()
        ds.update(log)
        
        ds['Trigger'] = Var( [int(x) for x in ds['Trigger']] ) # make sure Trigger is type Var
        
        # add subject label
        ds['Subject'] = Factor([subject], rep=ds.n_cases, random=True)
        
        # fix Brit spelling and 'spread(s)'
        # ds = ds[ds['Stimulus'].isnot('spread', 'spreads')]
        brit = {'favour':'favor', 'favours':'favors', 'favoured':'favored', 
                'honour':'honor', 'honours':'honors', 'honoured':'honored'}
        for w in brit:
            try:
                i = ds['Stimulus'].index(w)
                ds['Stimulus'][i] = brit[w]
            except KeyError: pass 
            
        # find corresponding behavioral file
        if subject == None:
            return ds
        else:
            try:
                rtPath = os.path.join('/Volumes','BackUp','sufAmb_behavioral','combined',subject+'.txt')
                rt = load.tsv(rtPath)
            except IOError:
                raise IOError('Could not find behavioral file for %s.' % subject)
                return ds 
                    
        ds = ds[ds['Stimulus'].isin(rt['Word'])]
        rt = rt[rt['Word'].isin(ds['Stimulus'])]
        
        # IMPORTANT: use order of words in ds, sort rt to match this order
        wordMap = dict((w,i) for i,w in enumerate(list(ds['Stimulus']))) # note order of words in ds
        index = []
        for case in range(rt.n_cases): 
            w = rt[case]['Word']
            i = wordMap[w]
            index.append(i)
        rt['index'] = Var(index)
        rt.sort('index') # align rows of ds with rt
        
        ds.update(rt, replace=False, info=False)
        
        return ds

    def load_log(self, subject=None):

        self.set(subject=subject)

        # trial list
        path = self.get('log-file', fmatch=True)
        ds = load.tsv(path, names=['StimOrTrig','ScreenType','BeforeOrAfter',
                                   'AbsTime','PracOrExper','TrialNum',
                                   'Condition', 'Stimulus','Trigger',
                                   'WordOrNonword'], start_tag='START',
                                   ignore_missing=True)
        
        # filter lines        
        # non-trigger screens                   
        index = ds['StimOrTrig'].isnot('STIM')
        # intermissions
        index = logical_and(index, ds['BeforeOrAfter'].isnot('', 'Tag'))
        # nonwords 
        index = logical_and(index, ds['BeforeOrAfter'].isnot('0'))
        # button press response -- ignore this for later version
        index = logical_and(index, ds['BeforeOrAfter'].isnot('128'))                 
                               
        ds = ds[index]
        
        del ds['Condition']

        return ds

# don't print a bunch of info
set_log_level('warning')

# create instance upon importing module
if socket.gethostname() == 'silversurfer.linguistics.fas.nyu.edu':
    e = SufAmb('/Volumes/BackUp/sufAmb')  # data is stored on silversurfer
else:
    print "Trying to connect to data drive."
    e = SufAmb('/Volumes/BackUp/sufAmb')

        
# Number of good trials per condition
# total: table.frequencies(Y='accept',ds=e.load_selected_events(reject='keep'))
# by condition: # print table.frequencies(Y='Condition',ds=e.load_selected_events(reject='keep').sub("accept==True"))
            
#e.set_inv(ori, depth, reg, snr, method, pick_normal)
#e.load_inv(fiff)

# parc=by lobes, masked=whole brain with medial wall removed
# pmin = tfc for thresh free
# mask = PALS_B12_Lobes
# e.make_report('TypeXDominanceCategory', parc= 'PALS_B12_Lobes', pmin=0.05, tstart=0.0, tstop=0.45, samples=30)

# if error, check that all subjects have the same source space
# for subject in e:
#     src = e.load_src()
#     print subject, src[0]['vertno']

