__author__ = 'Andrew Jamison'
# Adapted from Tal Linzen's LexVar class (https://github.com/TalLinzen/lexvars)

# TODO: compute these variables
# form_H_InfFam, form_H_affixes, NV_H_InfFam, NV_H_affixes, NV_stem_H
# NVAdj_H_InfFam, NVAdj_H_affixes
# Delta_H for all of the above, except NV_stem_H
# V_H ratio, Delta_V_H ratio for all above
# H_DerFam
# NV Synset count, H for types and types+tokens
# SUBTLEX, OLD20, AoA, transitivity, denominal/deverbal

import __builtin__
import collections
import os

import numpy as np
from celex import Celex
from corpus import Corpus

home = os.path.expanduser("~")
path_to_ds = os.path.join(home, "Dropbox", "SufAmb", "all_subjects.csv")
corPath = os.path.join(home, "Dropbox", "corpora")
elpPath = os.path.join(corPath, "ELP", "ELPfull.csv")
slxPath = os.path.join(corPath, "SUBTLEX-US.csv")
aoaPath = os.path.join(corPath, "AoA.csv")

if 'clx' not in globals():
    path_to_clx = os.path.expanduser("~/Dropbox/corpora/CELEX_V2/english")
    clx = Celex(path_to_clx)
    clx.load_lemmas()
    clx.load_wordforms()
    clx.map_lemmas_to_wordforms()
if 'elp' not in globals():
    elp = Corpus(elpPath)
if 'slx' not in globals():
    slx = Corpus(slxPath)
if 'aoa' not in globals():
    aoa = Corpus(aoaPath)
if 'wn' not in globals():
    from nltk.corpus import wordnet as wn


class LexVars(Corpus):

    def __init__(self, path, item_name='Word'):
        super(LexVars, self).__init__(path, item_name)
        self.debug = True  # print verbose messages by default

    def _warning_msg(self, func_name, comparison):
        if self.debug:
            print "WARNING: the following items have a value of 'None' for '%s':\n %s" \
                % (func_name, self.compare_items(comparison, True)['not_in_comparison'])
        else:
            print "WARNING: at least one item has a value of 'None' for '%s!\n" \
                  "Set 'debug' to 'True' to see the list of items." % func_name

    def lemma_headwords(self):
        """For each word returns the Celex lemma headword.
        """
        new_var = 'lemma_headword'
        lemma_heads = [clx._lemmas[i]['Head'] for i in xrange(len(clx._lemmas))]
        has_item = self.compare_items(lemma_heads)
        new_column = []
        if False in has_item:
            self._warning_msg('lemma_headword', lemma_heads)
        for record, exists in zip(self._dict, has_item):
            if exists:
                lemma_id = clx.wordform_lookup(record)[0].IdNumLemma
                lemma_head = clx.lemma_by_id(lemma_id).Head
            else:
                lemma_head = None
            new_column.append(lemma_head)
        self._append_column(new_column, new_var)

    def wordnet_synsets(self):
        """Number of WordNet "synsets" (roughly, senses) for the given word.
        This variable collapses across different parts of speech, meanings,
        etc.
        """
        new_var = "n_synsets"
        wn_lemmas = [l.encode('ascii') for l in wn.all_lemma_names()]
        has_item = self.compare_items(wn_lemmas)
        new_column = []
        if False in has_item:
            self._warning_msg('wordnet_synsets', wn_lemmas)
        for record, exists in zip(self._dict, has_item):
            if exists:
                n_synsets = len(wn.synsets(record))
            else:
                n_synsets = None
            new_column.append(n_synsets)
        self._append_column(new_column, new_var)

    def synset_ratio(self):
        new_var = "vtn_synsets"
        wn_lemmas = [l.encode('ascii') for l in wn.all_lemma_names()]
        has_item = self.compare_items(wn_lemmas)
        new_column = []
        if False in has_item:
            self._warning_msg('synset_ratio', wn_lemmas)
        for record, exists in zip(self._dict, has_item):
            if exists:
                n_synsets = len(wn.synsets(record, 'n'))
                v_synsets = len(wn.synsets(record, 'v'))
                vtn_synsets = float(v_synsets) / (n_synsets + v_synsets)
            else:
                vtn_synsets = None
            new_column.append(vtn_synsets)
        self._append_column(new_column, new_var)

    def age_of_acquisition(self):
        new_var = "AoA"
        has_item = self.compare_items(aoa)
        new_column1, new_column2 = [], []
        if False in has_item:
            self._warning_msg('age_of_acquisition', aoa)
        for record, exists in zip(self._dict, has_item):
            if exists:
                age = aoa[record]['AoA_Kup_lem']
                p_age = [record]['Perc_known_lem']
            else:
                age, p_age = None, None
            new_column1.append(age), new_column2.append(p_age)
        self._append_column(new_column1, new_var), self._append_column(new_column1, 'AoA_perc_known')

    def subtitles(self):
        new_var = "subtlx_log_fre"
        has_item = self.compare_items(slx)
        new_column1, new_column2 = [], []
        if False in has_item:
            self._warning_msg('subtitles', slx)
        for record, exists in zip(self._dict, has_item):
            if exists:
                fre = slx[record]['Lg10WF']
                cd = slx[record]['Lg10CD']
            else:
                fre, cd = None, None
            new_column1.append(fre), new_column2.append(cd)
        self._append_column(new_column1, new_var), self._append_column(new_column2, "subtlx_log_cd")

    def levenshtein_distance(self):
        var_names = ['OLD', 'OLDF', 'PLD', 'PLDF']
        new_vars = ['OLD', 'OLDF', 'PLD', 'PLDF']
        has_item = self.compare_items(elp)
        new_values = []
        if False in has_item:
            self._warning_msg('levenshtein_distance', elp)
        for record, exists in zip(self._dict, has_item):
            if exists:
                item_values = []
                for i in xrange(len(var_names)):
                    value = elp[record][var_names][i]
                    item_values.append(value)
            else:
                item_values = [None]*len(var_names)
            new_values.append(item_values)
        for record,


        for record in self._dict:
            self._dict[record]['OLD'] = elp[record]['OLD']
            self._dict[record]['OLDF'] = elp[record]['OLDF']
            self._dict[record]['PLD'] = elp[record]['PLD']
            self._dict[record]['PLDF'] = elp[record]['PLDF']

    def _pos_freq(self, lemma, pos):
        """Total frequency of the lemma when used as the given part of speech:

        >>> lv.pos_freq('wind', 'noun')
        3089
        """
        lemmas = clx.lemma_lookup(lemma)
        return sum(lemma.Cob for lemma in lemmas if lemma.ClassNum == pos)

    def _smoothed_log_ratio(self, a, b):
        return np.log2((a + 1.) / (b + 1.))

    def log_noun_to_verb_ratio(self, use_lemma=True):
        for record in self._dict:
            if use_lemma:
                item = self._dict[record]['lemma_headword']
            else:
                item = record
                raise ValueError("use_lemma must be True.")
            noun_freq = self._pos_freq(item, 'noun')
            verb_freq = self._pos_freq(item, 'verb')
            ratio = self._smoothed_log_ratio(noun_freq, verb_freq)
            self._dict[record]['log_ntv_ratio'] = ratio

    def inflectional_entropy(self, smooth=1, verbose=False, use_lemma=True):
        """This function collapses across all relevant lemmas, e.g. the noun
        "build" and the verb "build", or the various "wind" verbs.

        Caution: if there are two ways to express the same inflection, the
        function will treat them as the same cell in the inflection
        distribution (e.g. "hanged" and "hung"). Probably worth adding this
        as an option in a future version.

        This function supports the following three types of inflectional
        entropy, but there are many more ways to carve up the various
        inflections.

        Paradigm 1: separate_bare

        bare forms are separated into nominal and verbal, but the
        verbal bare form is not further differentiated between present
        plural agreeing form and infinitive

        ache (singular), aches (plural), ache (verb -- infinitive,
        present tense except third singular),
        aches (3rd singular present),
        aching (participle), ached (past tense),
        ached (participle -- passive and past_tense)

        Paradigm 2: collapsed_bare

        Same as separate_bare but collapsing across bare forms:

        ache (singular noun and all bare verbal forms --
        so all forms with no overt inflection), aches (plural),
        aches (3rd singular present), aching (participle),
        ached (past tense), ached (participles)

        Paradigm 3: no_bare

        Same as collapsed_bare, only without bare form:

        aches (plural), aches (3rd singular present),
        aching (participle), ached (past tense), ached (participles)
        """

        for record in self._dict:
            if use_lemma:
                item = self._dict[record]['lemma_headword']
            else:
                item = record
                raise ValueError("use_lemma must be True.")
            clx_lemmas = clx.lemma_lookup(item)
            # Use __builtin__ here in case sum is overshadowed by numpy
            all_wordforms = __builtin__.sum((clx.lemma_to_wordforms(clx_lemma)
                                             for clx_lemma in clx_lemmas), [])

            counter = collections.Counter()

            for wf in all_wordforms:
                infl = wf.FlectType
                freq = wf.Cob
                if (infl[0] == 'present_tense' and infl[1] != '3rd_person_verb'
                    or infl[0] == 'infinitive'):
                    counter['bare_verb'] += freq
                if infl[0] == 'singular':
                    counter['bare_noun'] += freq
                if infl[0] == 'plural':
                    counter['noun_plural'] += freq
                if infl[0] == 'past_tense':
                    counter['past_tense'] += freq
                if infl == ['positive']:
                    counter['positive'] += freq
                if infl == ['comparative']:
                    counter['comparative'] += freq
                if infl == ['superlative']:
                    counter['superlative'] += freq
                if infl == ['headword_form']:
                    counter['headword_form'] += freq
                if infl == ['present_tense', '3rd_person_verb', 'singular']:
                    counter['third_sg'] += freq
                if infl == ['participle', 'present_tense']:
                    counter['part_ing'] += freq
                if infl == ['participle', 'past_tense']:
                    counter['part_ed'] += freq

            common = ['noun_plural', 'third_sg', 'part_ing', 'part_ed',
                      'past_tense', 'comparative', 'superlative']
            bare = ['bare_noun', 'bare_verb', 'positive', 'headword_form']
            common_freqs = [counter[i] for i in common if i in counter]
            bare_freqs = [counter[i] for i in bare if i in counter]

            if verbose:
                print counter

            self._dict[record]['infl_ent_separate_bare'] = self.entropy(bare_freqs + common_freqs, smooth)
            self._dict[record]['infl_ent_collapsed_bare'] = self.entropy([sum(bare_freqs)] + common_freqs, smooth)
            self._dict[record]['infl_ent_no_bare'] = self.entropy(common_freqs, smooth)

    def derivational_family_size(self, use_lemma=True):

        by_morpheme = {}

        for lemma in clx._lemmas:
            if len(lemma['Parses']) < 1:
                continue  # no parses listed in Celex
            decomposition = lemma['Parses'][0]['Imm']  # use most common ([0]) parse
            morphemes = decomposition.split('+')
            for morpheme in morphemes:
                if morpheme in self._dict.keys():
                    by_morpheme.setdefault(morpheme, set()).add(lemma['Head'])

        for record in self._dict:
            if use_lemma:
                item = self._dict[record]['lemma_headword']
            else:
                item = record
                raise ValueError("use_lemma must be True.")
            decompositions = by_morpheme.get(item)
            if decompositions is not None:
                self._dict[record]['derivational_family_size'] = len(decompositions)
            else:
                self._dict[record]['derivational_family_size'] = 0

    def derivational_family_entropy(self, use_lemma=True):

        by_morpheme = {}

        for lemma in clx._lemmas:
            if len(lemma['Parses']) < 1:
                continue
            decomposition = lemma['Parses'][0]['Imm']
            morphemes = decomposition.split('+')
            if morphemes[0] in self._dict.keys():
                by_morpheme.setdefault(morphemes[0], list()).append(lemma)

        for record in self._dict:
            if use_lemma:
                item = self._dict[record]['lemma_headword']
            else:
                item = record
                raise ValueError("use_lemma must be True.")
            derived = by_morpheme.get(item)
            freqs = [x['Cob'] for x in derived]
            self._dict[record]['derivational_entropy'] = self.entropy(freqs)

    def entropy(self, freq_vec, smoothing_constant=1):
        """This flat smoothing is an OK default but probably not the best idea:
        might be better to back off to the average distribution for the
        relevant paradigm (e.g. if singular forms are generally twice as likely
        as plural ones, it's better to use [2, 1] as the "prior" instead of
        [1, 1]).
        """
        vec = np.asarray(freq_vec, float) + smoothing_constant
        if sum(vec) == 0:
            return -1
        probs = vec / sum(vec)
        # Make sure we're not taking the log of 0 (by convention if p(x) = 0
        # then p(x) * log(p(x)) = 0 in the definition of entropy)
        probs[probs == 0] = 1
        return -np.sum(probs * np.log2(probs))


def debug():
    lex = LexVars(path_to_ds)
    lex._append_column([w['Stem'] for w in lex], 'lemma_headword')  # b/c class method gets some headwords wrong
    return lex
