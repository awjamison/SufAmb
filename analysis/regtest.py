'''
Created on Aug 17, 2014

@author: andrew
'''

from sufAmb6 import e
from regression import ols
from eelbrain import save

IVs = ['Ambiguity', 'WrdVerbyWeighted', 'WrdBiasWeighted']
covar = ['FamilyOrder', 'LetNum', 'SylNum', 'OrthNeighborhoodFre', 
         'UnconstrainedBigramFre', 'Log_DerivationalFamilySize', 'LogSynsetNum_Stem',
         'ParadigmSurprisal', 'LogFre']

for iv in IVs:

    reg = ols(e, iv, covar=covar, fac='Type')

    path = '/Volumes/BackUp/sufAmb/regression/ols_%s.pickled' % iv
    save.pickle(reg, path)