'''
Created on Aug 21, 2014

@author: andrew
'''

from sufAmb6 import e
from correlation2 import ols
from eelbrain import save

IVs = ['VerbGivenWord', 'Ambiguity', ]
covar = None

for iv in IVs:

    reg = ols(e, iv, covar=covar, fac='Type')

    path = '/Volumes/BackUp/sufAmb/regression/corr_%s.pickled' % iv
    save.pickle(reg, path)