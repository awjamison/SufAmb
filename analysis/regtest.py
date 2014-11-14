'''
Created on Aug 17, 2014

@author: andrew
'''

from sufamb_experiment import e
from regression import ols
from eelbrain import save

tests = {'stem': [['VerbGivenStem', 'e3SVerbGivenStemS'], ['Ambiguity_Stem', 'Ambiguity_StemS']],
         'stemS': [['VerbGivenStem', 'e3SVerbGivenStemS'], ['Ambiguity_Stem', 'Ambiguity_StemS']],
         'main': [['VerbGivenWord']]}
covar = ['FamilyOrder', 'LetNum', 'SylNum', 'UnconstrainedBigramFre', 'OrthNeighborhoodSize',
         'DerivationalFamilySize', 'SynsetNum_Stem', 'LogFre', 'ParadigmSurprisal']

for lev in tests:
    if lev == 'main':
        factor, level = None, 'all'
    else:
        factor, level = 'Type', lev
    for t in range(len(tests[lev])):
        IVs = covar + tests[lev][t]
        reg = ols(e, IVs, factor=factor, level=level)
        varnames = '_'.join(tests[lev][t])
        name = lev + '-' + varnames
        path = '/Volumes/BackUp/sufAmb/regression/ols_%s.pickled' % name
        save.pickle(reg, path)