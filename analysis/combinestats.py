'''
Created on Aug 18, 2014

@author: andrew
'''

from eelbrain import combine, load, save

dr = '/Volumes/Backup/sufAmb/pickled/'
IVs = ['VerbGivenWord_nocov', 'VerbGivenWord', 'Ambiguity', 'WrdVerbyWeighted', 'WrdBiasWeighted']

for v in IVs:
    fil = dr+'ols_'+v+'.pickled'
    data = load.unpickle(fil)
    for k in data.keys():
        if k == 'src~all_VerbGivenWord':
            data[k]
    save.pickle(data, fil)
        