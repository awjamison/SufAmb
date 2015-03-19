import os
import numpy as np
from eelbrain import load, Var, Factor, combine

logPath = os.path.join('/Volumes','BackUp','sufAmb_behavioral', 'raw')
masterPath = os.path.join('/Volumes','BackUp','sufAmb_stimuli','stimuli',
                          'stim_properties','master_list35_update6.txt')
masterRT = load.tsv(masterPath) # complete set of stimuli and properties

def load_logs(log_path=logPath):

    log_names = os.listdir(log_path)

    # remove other files in the directory
    if '.DS_Store' in log_names:
        log_names.remove('.DS_Store')
    if 'old' in log_names:
        log_names.remove('old')

    logs = []
    for s in log_names:
        path = os.path.join(logPath, s)
        log = load.tsv(path)
        log.info['name'] = log.name  # store name here b/c log.name gets reset
        # fix 'spread(s)'
        log = log[log['Stimulus'].isnot('spread', 'spreads')]
        if log.info['name'] == 'R0823_s1_stimuli_data.txt': # this subject doesn't have the same nonword stimuli
            log = log[log['Correct'].isnot('NotWord')]
        log.name = s
        logs.append(log)

    print 'Done reading log files.'
    return logs

def get_word_info(logs=load_logs(), master_rt=masterRT):
    """
    Adds word properties to the log file. Preserves order of presentation in the log file.
    """

    dss = []   
    for log in logs:
        
        ds = master_rt[master_rt['Word'].isin(log['Stimulus'])]  # look up words in master_rt ONLY IF in log

        wordMap = dict((w,i) for i,w in enumerate(list(log['Stimulus'])))  # create an index based on order presented
        index = [] # copy the index to ds
        for case in range(ds.n_cases):
            w = ds[case]['Word']
            i = wordMap[w]
            index.append(i)
        ds['index'] = Var(index)
        ds.sort('index')  # align rows of ds and log
        ds.update(log)  # concatenate ds and log
        orderrun = Var( [int(log.name.split('_')[1][1:])] * ds.n_cases ) 
        subject = Factor([log.name[:5]] * ds.n_cases)
        ds['OrderRun'] = orderrun  # order in which subject was run
        ds['Subject'] = subject  # first five characters give R number
        ds.name = log.name[:5] 
        dss.append(ds)
        print ds.name
        
    print 'Done adding word info.'
    
    return dss


def resp_accuracy(ds):
    """
    Note: R0850_s10 responses reversed; fixed manually in log file
    """
    
    responseKey = {'1': 'NotWord', '2': 'IsWord', 'TIMEOUT': None}
    
    IsCorrect = []
    wCorrect = []
    nwCorrect = []
    for row in range(ds.n_cases):
        if responseKey[ ds[row]['Button'] ] == ds[row]['Correct']:
            IsCorrect.append(1)
            if responseKey[ ds[row]['Button'] ] == 'IsWord':
                wCorrect.append(1)
            else:
                nwCorrect.append(1)
        else:
            IsCorrect.append(0)
            if responseKey[ ds[row]['Button'] ] == 'IsWord':
                wCorrect.append(0)
            else:
                nwCorrect.append(0)
    ds['IsCorrect'] = Var(IsCorrect)
    overall = float(sum(IsCorrect)) / len(IsCorrect)
    wrds = float(sum(wCorrect)) / len(wCorrect)
    nwrds = float(sum(nwCorrect)) / len(nwCorrect)
    print "Overall percent correct: %f" % overall
    print "Word percent correct: %f" % wrds
    print "Nonword percent correct: %f" % nwrds
    
    return ds
   
                
def resp_latency(ds, lower_cutoff=0.25, upper_cutoff=1.5, sd=4.5):
    """
    Throw out RTs that are too fast or too short. First filter by cutoff, then by sd
    WARNING: must be same order as original log file!
    """
       
    RTs = [float(x) for x in ds['RT']]
    total = len([float(x) for x in ds['RT']])
    # record # of long RTs
    badRTs = len([x for x in RTs if x < lower_cutoff or x > upper_cutoff])
    # eliminate timeouts (> 4 seconds) for outlier calculation
    RTs = [x for x in RTs if x >= 0]
    # eliminate long responses for outlier calculation
    RTs = [x for x in RTs if x >= lower_cutoff and x <= upper_cutoff]
    # descriptive statistics 
    meanRT = np.mean(RTs)
    stdRT = np.std(RTs)
    upper = meanRT + sd * stdRT
    lower = meanRT - sd * stdRT
    outliers = len([x for x in RTs if x > upper or x < lower])
    percentage = float(total - badRTs - outliers) / total
    print 'Mean: %f, SD: %f, Long RTs: %i, Outliers: %i' % (meanRT, stdRT, badRTs, outliers)
    print 'Percent with good latency: %f' % percentage
    
    GoodLatency = []
    PrecedingRT = [str(meanRT)] # first element of PrecedingRT = subject's mean RT
    Preceding2RT = [str(meanRT), str(meanRT)] # first two elements of Preceding2RT = subject's mean RT
    Preceding3RT = [str(meanRT), str(meanRT), str(meanRT)]
    for row in range(ds.n_cases):
        
        if float(ds[row]['RT']) > upper or float(ds[row]['RT']) < lower:
            GoodLatency.append(0)
        else:
            GoodLatency.append(1)
        
        if row > 0: # first row has no preceding RT
            if float(ds['RT'][row-1]) < 0 :
                PrecedingRT.append('4.0')  # set timeouts = 4 seconds
            else:
                PrecedingRT.append(ds['RT'][row-1])   
            
        if row > 1: # for Preceding2RT
            if float(ds['RT'][row-2]) < 0 :
                Preceding2RT.append('4.0')  # set timeouts = 4 seconds
            else:
                Preceding2RT.append(ds['RT'][row-2])
                
        if row > 2: # for Preceding3RT
            if float(ds['RT'][row-3]) < 0 :
                Preceding3RT.append('4.0')  # set timeouts = 4 seconds
            else:
                Preceding3RT.append(ds['RT'][row-3])

    ds['PrecedingRT'] = Factor(PrecedingRT)
    ds['Preceding2RT'] = Factor(Preceding2RT)
    ds['Preceding3RT'] = Factor(Preceding3RT)
    ds['GoodLatency'] = Var(GoodLatency) 
   
    return ds


def resp_good(ds):
    """
    Number of good responses based on latency and accuracy
    """
    
    GoodResponse = []
    for row in range(ds.n_cases):
        if ds[row]['IsCorrect'] == 1 and ds[row]['GoodLatency'] == 1:
            GoodResponse.append(1)
        else:
            GoodResponse.append(0)
    ds['GoodResponse'] = Var(GoodResponse)
    goodTotal = float(sum(GoodResponse)) / len(GoodResponse)
    print 'Percent of good response: %f' % goodTotal
    
    return ds
     

def item_order(ds):
    """
    Rank order in which inflections of a common stem were seen.
    """
    if ds.has_key('index') == False:
        raise KeyError('Dataset does not contain an index!')
    if list(ds['index']) != range(ds.n_cases): # if the count starts at 0
        if list(ds['index']) != range(ds.n_cases) + 1: # if the count starts at 1
            raise ValueError('Dataset is not ordered according to index!')
    
    # create a dictionary that keeps track of the index for the next occurrence of the inflection
    n = [1]*ds.n_cases
    s = set(ds['Stem'])
    count = zip(s,n)
    count = dict(count)
    
    FamilyOrder = []
    for s in ds['Stem']:
        FamilyOrder.append(count[s])
        count[s] += 1
    ds['FamilyOrder'] = Var(FamilyOrder)
    
    return ds
            
            
def resp_filter(dss, threshold=0.0):
    """
    Display response statistics for each subject
    """
    
    updated = []
    
    for ds in dss:
        print '----------%s----------' % ds.name
        ds = resp_latency(ds)
        ds = resp_accuracy(ds)
        ds = resp_good(ds)
        ds = item_order(ds)
        if float(sum(ds['GoodResponse'])) / len(ds['GoodResponse']) > threshold:
            updated.append(ds)
            print 'ACCEPTED'
        else:
            print 'REJECTED'
    
    return updated
    

outFold = os.path.join('/Volumes','BackUp','sufAmb_behavioral','combined')


def write_logs(dss, output='concatenated', outFold=outFold, includeNonwords = False):
    """
    'concatenated': one output file
    'separated': multiple output files
    """
    
    if output == 'separated':
        for ds in dss:
            if includeNonwords == False:
                ds = ds[ds['RealWord'].isnot(0)]
            outPath = os.path.join(outFold, ds.name)
            ds.save_txt(outPath)
        print 'Done writing separate log files'
    elif output == 'concatenated':
        combined = combine(dss)
        if includeNonwords == False:
            combined = combined[combined['RealWord'].isnot(0)]
        outPath = os.path.join(outFold, 'all_subjects')
        combined.save_txt(outPath)
        print 'Done writing concatenated file'
        