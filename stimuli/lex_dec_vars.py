__author__ = 'Andrew Jamison'
# much of the code is adapted from Tal Linzen (https://github.com/TalLinzen)

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
        self._lemid_lookup = _make_LemID_lookup()
        self._flect_lookup = _make_Flect_lookup()

    def _make_LemID_lookup(self):
        if 'clx-lemmas' not in self.db:
            self.read_clx()
        LemID_lookup = {}
        for wf in self.db['clx-wordforms']:
            LemID_lookup.setdefault(wf['Word'], set()).add(wf['IdNumLemma'])
        return LemID_lookup

    def _make_Flect_lookup(self):
        if 'clx-lemmas' not in self.db:
            self.read_clx()
        Flect_lookup = {}
        for wf in self.db['clx-wordforms']:
            Flect_lookup.setdefault(wf['IdNumLemma'], set()).add(wf['Word'])
        return Flect_lookup

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
                record['deriv_family_size'] = 0

    def right_derivational_family_entropy(self, include_multiword=True):
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
            if morphemes[0] in self.words:
                by_morpheme.setdefault(morphemes[0], list()).append(lemma)

        for record in self.db['exp']:
            derived = by_morpheme.get(record['Word'])
            if derived is not None:
                freqs = [x['Cob'] for x in derived]
                record['right_deriv_entropy'] = self.entropy(freqs)
            else:
                record['right_deriv_entropy'] = 0

    def entropy(self, vec, smoothing_constant=1):
        """H(A) == sum_x( -p(x_i|a)*log(p(x_i|a) ) where x is the
        cardinality of X, a is the cardinality of A, and X is in A.
        """

        freqs = [float(x) + smoothing_constant for x in vec]
        if sum(freqs) == 0:
            return -1
        probs = [f / sum(freqs) for f in freqs]
        s = 0
        for p in probs:
            s -= 0 if p == 0 else p * math.log(p)
        s /= math.log(2)
        return s



# for testing
def debug():
    l = LexDecVars(expPath, debug=True)

if __name__ == '__main__':
    debug()