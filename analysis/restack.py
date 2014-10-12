'''
Created on Aug 18, 2014

@author: andrew
'''

from eelbrain import Dataset, Factor, combine, load, save

#===============================================================================
# current format: [subject1, stem, stemS, stemEd],
#                 [subject2, stem, stemS, stemEd],
#                 ...
#
# new format: [subject1, stem],
#             [subject1, stemS], 
#             ...
#             [subject2, stem], 
#             ...
#===============================================================================

dr = '/Volumes/Backup/sufAmb/pickled/'
IVs = ['VerbGivenWord_nocov', 'VerbGivenWord', 'Ambiguity', 'WrdVerbyWeighted', 'WrdBiasWeighted']

for v in IVs:
    fil = dr+'ols_'+v+'.pickled'
    data = load.unpickle(fil)
    c1, c3 = [], []
    c2 = ['all', 'stem', 'stemS', 'stemEd']*data.n_cases
    for s in range(data.n_cases):
        c1.extend( [ data['subject'][s] ]*4 )
        c3.extend( [ data['all'][s], data['stem'][s], data['stemS'][s], data['stemEd'][s] ] )
        
    c1 = Factor(c1)
    c2 = Factor(c2)
    c3 = combine(c3)
        
    newds = Dataset(('subject',c1), ('Type',c2), ('beta',c3), info=data.info)
    
    save.pickle(newds, fil)