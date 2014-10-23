__author__ = 'Andrew Jamison'

import math
from corpus import *
from nltk.corpus import wordnet as wn

# TODO: compute these variables
# form_H_InfFam, form_H_affixes, NV_H_InfFam, NV_H_affixes, NV_stem_H
# NVAdj_H_InfFam, NVAdj_H_affixes
# Delta_H for all of the above, except NV_stem_H
# V_H ratio, Delta_V_H ratio for all above
# H_DerFam
# NV Synset count, H for types and types+tokens
# SUBTLEX, OLD20, AoA, transitivity, denominal/deverbal


class LexDecVars(Corpus):

    def __init__(self, path, item_name='Word', debug=False):
        super(LexDecVars, self).__init__()  # to inherit data attributes
        self.read(path, 'exp', item_name)
        self.words = tuple(w[item_name] for w in self.db['exp'])
        self.debug = debug

    def derivational_family_size(self, include_multiword=True):
        # include_multiword: if this option is set to false, only
        # one-word lemmas are considered. For example, "sleep in" and
        # "beauty-sleep" are ignored, but "oversleep" and "sleepwalk" are
        # included. This is really arbitrary. Unfortunately.
        if 'clx-lemmas' not in self.db:
            self.read_clx()

        by_morpheme = {}

        for lemma in self.db['clx-lemmas']:
            if not include_multiword and ('-' in lemma['Head'] or
            ' ' in lemma['Head']):
                continue
            decomposition = lemma['Imm']
            morphemes = decomposition.split('+')
            for morpheme in morphemes:
                if morpheme in self.words:
                    by_morpheme.setdefault(morpheme, set()).add(lemma['Head'])

        for record in self.db['exp']:
            decompositions = by_morpheme.get(record['Word'])
            if decompositions is not None:
                record['deriv_family_size'] = len(decompositions)
                if self.debug:
                    record['deriv_family_size'] = decompositions
            else:
                record['deriv_family_size'] = None

# for testing
def debug():
    l = LexDecVars(expPath, debug=True)

if __name__ == '__main__':
    debug()