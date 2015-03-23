import os
import socket
from mne import set_log_level
from numpy import logical_and

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
    """Class for Andrew's sufAmb experiment.
    """

    groups = {'test':['R0053', 'R0606']}
    
    # R0632: weird sensor space grand average
    # R0689: excessive noise
    # R0851: excessive noise, metal in clothing
    exclude = {'subject':['R0632', 'R0689', 'R0851']}
    
    bad_channels = bad_channels
    
    epochs = {'100-600': dict(sel="BeforeOrAfter != 128", tmin=-0.15, tmax=0.6),
              '150-450': dict(sel="BeforeOrAfter != 128", tmin=-0.15, tmax=0.45),
              '150-450bl': dict(sel_epoch='150-450', tmin=-0.15, tmax=0),
              '150-450_behav_fltr': dict(sel_epoch='150-450', sel="GoodResponse == 1", tmin=-0.15, tmax=0.45),
              'cov': dict(sel_epoch='150-450', tmin=-0.15, tmax=0)
              }

    _eog_sns = ['MEG 143', 'MEG 151']
    
    _values = {'experiment': ('sufAmb'),
               'log-dir': os.path.join('{meg-dir}', 'log'),
               'log-file': os.path.join('{log-dir}', 
                                        '{subject}_s*_stimuli_log.txt'),
               }
               
    defaults = {'cov': 'bestreg',
                 'raw': '1-40',
                 'epoch': '150-450_behav_fltr',
                 'rej': 'man',
                 'model': 'Condition',
                 'parc': 'SufAmb_aparc',
                 'src': 'ico-4'
                 }
    
    parcs = ['', 'frontal_temporal_parietal', 'SufAmb_aparc', 'SufAmb_Brodmann']

    def label_events(self, ds):
        """ Wrapper for MneExperiment.label_events.
        Combines ds of stimuli variables/behavioral statistics with the event file from ~/raw.
        This means epoch files will contain all necessary information for statistical tests, which
        obviates merging the epoch files and stat files later.
        """

        ds = MneExperiment.label_events(self, ds)

        ds = ds[ds['trigger'].isnot(128)]
        # TODO keep button press triggers. Figure out which to get rid of in ~/raw/evts

        log = self.load_log()

        ds.update(log)  # combine

        return ds

    def load_log(self):
        """Loads the log of what stimulus screen was presented and combines the log with
        stimuli variables and behavioral statistics.
        """

        subject = self.get('subject')

        # load the log file
        path = self.get('log-file', fmatch=True)
        log = load.tsv(path)
        log = log[log['BeforeOrAfter'].isnot(128)]  # remove triggers to button press
        # TODO keep button press triggers. Requires mangling rt to repeat each row

        # load the processed behavioral file + stimulus variables
        rt_path = os.path.join('/Volumes', 'BackUp', 'sufAmb_behavioral', 'combined', subject+'.txt')
        try:
            rt = load.tsv(rt_path)
        except IOError:
            raise IOError('Could not find behavioral file for %s.' % subject)

        # combine the log file and behavioral/stim variable files
        word_map = dict((w, i) for i, w in enumerate(list(log['Stimulus'])))  # note order of words in log
        index = []
        for case in range(rt.n_cases):
            w = rt[case]['Word']
            i = word_map[w]
            index.append(i)
        rt['index'] = Var(index)
        rt.sort('index')  # align rows of log with rt
        log.update(rt, replace=False, info=False)  # combine

        log = log[log['RealWord'] == 1]  # event file currently does not contain nonwords

        log.sort('AbsTime')  # make sure log file is sorted by time

        return log

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