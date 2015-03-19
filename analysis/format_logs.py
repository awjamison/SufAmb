__author__ = 'andrew'

import os
from eelbrain import load

def reformat_logs():

    # first, format the logs in /meg directory
    meg_path = os.path.join('/Volumes', 'Backup', 'sufAmb', 'meg')
    subjects = os.listdir(meg_path)
    if '.DS_Store' in subjects:
        subjects.remove('.DS_Store')
    for s in subjects:
        log_path = os.path.join(meg_path, s, 'log', 'old')
        file_names = os.listdir(log_path)
        if '.DS_Store' in file_names:
            file_names.remove('.DS_Store')
        for f in file_names:
            f_path = os.path.join(log_path, f)
            if 'data' in f:
                # 'data' file
                ds = load.tsv(f_path, names=['Key','Button','RT',
                                    'Abs_RT','Shown','Tag',
                                    'Experiment', 'Trial','Condition',
                                    'Stimulus', 'Trigger', 'Correct'],
                                    start_tag='START',
                                    ignore_missing=True)
                # filter unanalyzed screens
                ds = ds[ds['Shown'].isnot('Paragraph')]
                ds = ds[ds['Experiment'].isnot('practice')]
                ds = ds[ds['Stimulus'].isnot('')]
                del ds['Condition']
                if f == 'R0823_s1_stimuli_data.txt':  # change british spelling for this subject
                    brit = {'favour':'favor', 'favours':'favors', 'favoured':'favored',
                            'honour':'honor', 'honours':'honors', 'honoured':'honored'}
                    for w in brit:
                        i = ds['Stimulus'].index(w)
                        ds['Stimulus'][i] = brit[w]
                # remove 'spread' and 'spreads' if they are in ds
                ds = ds[ds['Stimulus'].isnot('spread', 'spreads')]
            elif 'log' in f:
                # 'log' files
                ds = load.tsv(f_path, names=['StimOrTrig','ScreenType','BeforeOrAfter',
                                    'AbsTime','PracOrExper','TrialNum',
                                    'Condition', 'Stimulus','Trigger',
                                    'WordOrNonword'], start_tag='START',
                                    ignore_missing=True)
                # filter unanlayzed screens
                ds = ds[ds['StimOrTrig'].isnot('STIM')]
                ds = ds[ds['BeforeOrAfter'].isnot('', 'Tag')]  # intermissions
                if f == 'R0823_s1_stimuli_log.txt':  # change british spelling for this subject
                    brit = {'favour':'favor', 'favours':'favors', 'favoured':'favored',
                            'honour':'honor', 'honours':'honors', 'honoured':'honored'}
                    for w in brit:
                        i = ds['Stimulus'].index(w)
                        ds['Stimulus'][i] = brit[w]
                # remove 'spread' and 'spreads' if they are in ds
                ds = ds[ds['Stimulus'].isnot('spread', 'spreads')]
            else:
                raise IOError('Could not find log file.')
            ds.save_txt(os.path.join(meg_path, s, 'log', f))

    # second, format the logs in /SufAmb_behavioral directory
    rt_path = os.path.join('/Volumes', 'Backup', 'sufAmb_behavioral', 'raw', 'old')
    file_names = os.listdir(rt_path)
    if '.DS_Store' in file_names:
        file_names.remove('.DS_Store')
    for f in file_names:
        f_path = os.path.join(rt_path, f)
        ds = load.tsv(f_path, names=['Key','Button','RT',
                            'Abs_RT','Shown','Tag',
                            'Experiment', 'Trial','Condition',
                            'Stimulus', 'Trigger', 'Correct'],
                            start_tag='START',
                            ignore_missing=True)
        # unanalyzed screens
        ds = ds[ds['Shown'].isnot('Paragraph')]
        ds = ds[ds['Experiment'].isnot('practice')]
        ds = ds[ds['Stimulus'].isnot('')]
        del ds['Condition']
        if f == 'R0823_s1_stimuli_data.txt':  # change british spelling for this subject
            brit = {'favour':'favor', 'favours':'favors', 'favoured':'favored',
                    'honour':'honor', 'honours':'honors', 'honoured':'honored'}
            for w in brit:
                i = ds['Stimulus'].index(w)
                ds['Stimulus'][i] = brit[w]
        # remove 'spread' and 'spreads' if they are in ds
        ds = ds[ds['Stimulus'].isnot('spread', 'spreads')]
        ds.save_txt(os.path.join('/Volumes', 'Backup', 'sufAmb_behavioral', 'raw', f))

if __name__ == '__main__':
    reformat_logs()
    print 'Done'

path = os.path.join('/Volumes', 'Backup', 'sufAmb_behavioral', 'raw', 'R0823_s1_stimuli_data.txt')
ds = load.tsv(path)
