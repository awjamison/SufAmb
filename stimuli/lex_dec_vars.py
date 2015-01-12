__author__ = 'Andrew Jamison'
# Adapted from Tal Linzen's LexVar class (https://github.com/TalLinzen/lexvars)

# TODO: add documentation

import __builtin__
import collections
import os

import numpy as np
from celex import Celex
from corpus import Corpus
from brit_spelling import change_spelling

home = os.path.expanduser("~")
path_to_ds = os.path.join(home, "Dropbox", "SufAmb", "master_list35_update4.csv")
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
        self.copy_columns(aoa, ['AoA_Kup_lem', 'Perc_known_lem'])

    def subtitles(self):
        self.copy_columns(slx, ['Lg10WF', 'Lg10CD'])

    def levenshtein_distance(self):
        self.copy_columns(elp, ['OLD', 'OLDF', 'PLD', 'PLDF'])

    def _pos_freq(self, lemma, pos):
        """Total frequency of the lemma when used as the given part of speech:

        >>> lv.pos_freq('wind', 'noun')
        3089
        """
        lemmas = clx.lemma_lookup(lemma)
        return sum(lemma.Cob for lemma in lemmas if lemma.ClassNum == pos)

    def _smoothed_log_ratio(self, a, b):
        return np.log2((a + 1.) / (b + 1.))

    def log_noun_to_verb_ratio(self):
        new_var = "ntv_ratio"
        lemma_heads = self._dict['lemma_headword']
        has_item = self.compare_items(lemma_heads)
        new_column = []
        if False in has_item:
            self._warning_msg('log_noun_to_verb_ratio', lemma_heads)
        for record, exists in zip(self._dict, has_item):
            if exists:
                item = self._dict[record]['lemma_headword']
                noun_freq = self._pos_freq(item, 'noun')
                verb_freq = self._pos_freq(item, 'verb')
                ratio = self._smoothed_log_ratio(noun_freq, verb_freq)
            else:
                ratio = None
            new_column.append(ratio)
        self._append_column(new_column, new_var)

    def get_subtlex_freq(self, word, part_of_speech=None, log_freq=False, warning=False):
        """Returns count per million from SUBTLEX.

        When part of speech is specified in the optional second argument,
        only the count for that part of speech is returned. SUBTLEX doesn't
        provide enough information to accurately determine the frequency
        distribution of a word with more than two parts of speech, so it is
        advisable to leave the argument unspecified in such cases unless
        one has reason to believe that all parts of speech after the second
        contribute negligibly to the total frequency count for the word.
        """
        if log_freq:
            total_freq = float(slx[word]['Lg10WF'])  # log10 count per million
        else:
            total_freq = float(slx[word]['SUBTLWF'])  # count per million
        perc_dom = float(slx[word]['Percentage_dom_PoS'])
        categories = slx[word]['All_PoS_SUBTLEX'].split('.')
        if part_of_speech is None:
            return total_freq
        if part_of_speech == categories[0]:
            return total_freq * perc_dom
        if len(categories) > 1:
            if part_of_speech == categories[1]:
                return total_freq * (1 - perc_dom)
        if warning:
            print "WARNING: unable to calculate part of speech frequency because \n" \
                  "%s is not one of the word's two most frequent parts of speech.\n" \
                  "'0' returned instead." % part_of_speech
        return 0

    def subtlex_verb_ratio(self, log_freq=False):
        new_var = "slx_v_ratio"
        has_item = self.compare_items(slx)
        new_column = []
        if False in has_item:
            self._warning_msg('subtlex_verb_ratio', slx)
        for record, exists in zip(self._dict, has_item):
            if exists:
                total_freq = self.get_subtlex_freq(record, log_freq=log_freq)
                verb_freq = self.get_subtlex_freq(record, 'Verb', log_freq=log_freq)
                ratio = verb_freq / float(total_freq)
            else:
                ratio = None
            new_column.append(ratio)
        self._append_column(new_column, new_var)

    def inflectional_entropy(self, use_subtlex=True, smooth=1, verbose=False):
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
        get_fieldnames = True
        warn_msg = False
        for record in self._dict:
            clx_lemmas = clx.lemma_lookup(self._dict[record]['lemma_headword'])
            # Use __builtin__ here in case sum is overshadowed by numpy
            all_wordforms = __builtin__.sum((clx.lemma_to_wordforms(clx_lemma)
                                             for clx_lemma in clx_lemmas), [])

            counter = collections.Counter()

            for wf in all_wordforms:
                infl = wf.FlectType
                clx_freq = wf.Cob  # huge rounding error with freq/million, so just use raw count
                if use_subtlex:
                    if False in [w.Word in slx for w in all_wordforms]:
                        # not all members of an inflectional family could be found in SUBTLEX.
                        # for this family use CELEX instead of SUBTLEX to calculate entropy.
                        all_flects_in_subtlex = False
                        warn_msg = True
                    else:
                        all_flects_in_subtlex = True

                if infl[0] == 'present_tense' and infl[1] != '3rd_person_verb' or infl[0] == 'infinitive':
                    if use_subtlex and all_flects_in_subtlex:
                        counter['bare_verb'] = self.get_subtlex_freq(wf.Word, 'Verb')
                    else:
                        counter['bare_verb'] += clx_freq
                if infl[0] == 'singular':
                    if use_subtlex and all_flects_in_subtlex:
                        counter['bare_noun'] = self.get_subtlex_freq(wf.Word, 'Noun')
                    else:
                        counter['bare_noun'] += clx_freq
                if infl[0] == 'plural':
                    if use_subtlex and all_flects_in_subtlex:
                        counter['noun_plural'] = self.get_subtlex_freq(wf.Word, 'Noun')
                    else:
                        counter['noun_plural'] += clx_freq
                if infl == ['present_tense', '3rd_person_verb', 'singular']:
                    if use_subtlex and all_flects_in_subtlex:
                        counter['third_sg'] = self.get_subtlex_freq(wf.Word, 'Verb')
                    else:
                        counter['third_sg'] += clx_freq
                if infl == ['participle', 'present_tense']:
                    if use_subtlex and all_flects_in_subtlex:
                        counter['verb_ing'] = self.get_subtlex_freq(wf.Word, 'Verb')
                        counter['adj_ing'] = self.get_subtlex_freq(wf.Word, 'Adjective')
                    else:
                        counter['verb_ing'] += clx_freq
                # past participle and past tense are collapsed, so doesn't work for irregulars
                if infl == ['participle', 'past_tense'] or infl[0] == 'past_tense':
                    if use_subtlex and all_flects_in_subtlex:
                        counter['verb_ed'] = self.get_subtlex_freq(wf.Word, 'Verb')
                        counter['adj_ed'] = self.get_subtlex_freq(wf.Word, 'Adjective')
                    else:
                        counter['verb_ed'] += clx_freq

            counter = {k: counter[k] * 1000 for k in counter}  # after freqs are tallied, multiply all freqs by 1000

            common = ['noun_plural', 'third_sg', 'verb_ing', 'adj_ing', 'verb_ed', 'adj_ed']
            bare = ['bare_noun', 'bare_verb']
            # stemS | stemEd | stemIng | == common
            stemS = ['noun_plural', 'third_sg']
            stemEd = ['verb_ed', 'adj_ed']
            stemIng = ['verb_ing', 'adj_ing']
            nouns = ['bare_noun', 'noun_plural']
            verbs = ['bare_verb', 'third_sg', 'verb_ed', 'verb_ing']
            adjectives = ['adj_ed', 'adj_ing']

            # frequency counts
            f_common = [counter[i] for i in common if i in counter]
            f_bare = [counter[i] for i in bare if i in counter]
            f_stemS = [counter[i] for i in stemS if i in counter]
            f_stemEd = [counter[i] for i in stemEd if i in counter]
            f_stemIng = [counter[i] for i in stemIng if i in counter]
            f_Vstem = [counter['bare_verb']]
            f_Nstem = [counter['bare_noun']]
            f_VstemS = [counter['third_sg']]
            f_NstemS = [counter['noun_plural']]
            f_nouns = [counter[i] for i in nouns if i in counter]
            f_verbs = [counter[i] for i in verbs if i in counter]
            f_adjs = [counter[i] for i in adjectives if i in counter]

            if verbose:
                print counter

            # table of frequency distributions
            dist = [
                ('separate_bare', f_bare + f_common),
                ('collapsed_bare', [sum(f_bare)] + f_common),
                ('no_bare', f_common),
                ('wordforms', [sum(f_bare)] + [sum(f_stemS)] + [sum(f_stemEd)] + [sum(f_stemIng)]),
                ('affixed_wordforms', [sum(f_stemS)] + [sum(f_stemEd)] + [sum(f_stemIng)]),
                ('stem', f_bare),
                ('stemS', f_stemS),
                ('stemEd', f_stemEd),
                ('stemIng', f_stemIng),
                ('stemAndstemS_pos', f_bare + f_stemS),
                ('stemAndstemS_wordforms', [sum(f_bare)] + [sum(f_stemS)]),
                ('verbs', f_verbs),
                ('nouns', f_nouns),
                ('adjectives', f_adjs),
                ('collapsed_NV', [sum(f_nouns)] + [sum(f_verbs)]),
                ('collapsed_pos', [sum(f_nouns)] + [sum(f_verbs)] + [sum(f_adjs)])
            ]
            dist = collections.OrderedDict(dist)
            # calculate entropy measures from frequency distributions
            H = [('H_'+x, self.entropy(dist[x], smooth)) for x in dist]
            H = collections.OrderedDict(H)
            # calculate change in entropy from prior to posterior distribution
            delta_dist = [
                ('stem_pos', (dist['separate_bare'], dist['stem'])),
                ('stem_wordforms', (dist['wordforms'],  [sum(dist['stem'])])),
                ('Vstem_verbs', (dist['verbs'], f_Vstem)),
                ('stemS_all_pos', (dist['separate_bare'], dist['stemS'])),
                ('stemS_affixes_pos', (dist['no_bare'], dist['stemS'])),
                ('stemS_all_wordforms', (dist['wordforms'], [sum(dist['stemS'])])),
                ('stemS_affixes_wordforms', (dist['affixed_wordforms'], [sum(dist['stemS'])])),
                ('VstemS_verbs', (dist['verbs'], f_VstemS))
            ]
            deltaH = collections.OrderedDict()
            for measure in delta_dist:
                name = measure[0]
                prior_dist = measure[1][0]
                posterior_dist = measure[1][1]
                difference = []  # difference = {prior} - {posterior}
                for type_freq in prior_dist:
                    if type_freq not in posterior_dist:
                        difference.append(type_freq)
                deltaH['deltaH_'+name] = -self.entropy(difference, smooth)  # negative sign for entropy reduction
            ratioH = [
                ('verbs_all', H['H_verbs'] / float(H['H_separate_bare'])),
                ('deltaVstem_deltaNVstem', deltaH['deltaH_Vstem_verbs'] / float(deltaH['deltaH_stem_pos'])),
                ('deltaVstemS_deltaNVstemS', deltaH['deltaH_VstemS_verbs'] / float(deltaH['deltaH_stemS_all_pos']))
            ]
            ratioH = [('ratioH_'+x[0], x[1]) for x in ratioH]  # add 'ratioH' prefix
            ratioH = collections.OrderedDict(ratioH)
            lemma_total = float(sum(dist['wordforms']))
            affixes_total = float(sum(dist['affixed_wordforms']))
            verbs_total = float(sum(dist['verbs']))
            nouns_total = float(sum(dist['nouns']))
            ratioFre = [
                ('stem_lemma', sum(dist['stem']) / lemma_total),
                ('stemS_lemma', sum(dist['stemS']) / lemma_total),
                ('stemEd_lemma', sum(dist['stemEd']) / lemma_total),
                ('stemIng_lemma', sum(dist['stemIng']) / lemma_total),
                ('stemS_affixes', sum(dist['stemS']) / affixes_total),
                ('stemEd_affixes', sum(dist['stemEd']) / affixes_total),
                ('stemIng_affixes', sum(dist['stemIng']) / affixes_total),
                ('Vlemma_lemma', verbs_total / lemma_total),
                ('Nlemma_lemma', nouns_total / lemma_total)
            ]
            if verbs_total == 0:  # avoid dividing by zero
                ratioFre.append(('Vstem_Vlemma', 0))
                ratioFre.append(('VstemS_Vlemma', 0))
            else:
                ratioFre.append(('Vstem_Vlemma', f_Vstem[0] / verbs_total))
                ratioFre.append(('VstemS_Vlemma', f_VstemS[0] / verbs_total))
            if nouns_total == 0:
                ratioFre.append(('Nstem_Nlemma', 0))
                ratioFre.append(('NstemS_Nlemma', 0))
            else:
                ratioFre.append(('Nstem_Nlemma', f_Nstem[0] / nouns_total))
                ratioFre.append(('NstemS_Nlemma', f_NstemS[0] / nouns_total))
            ratioFre = [('ratioFre_'+x[0], x[1]) for x in ratioFre]  # add 'ratioFre' prefix
            ratioFre = collections.OrderedDict(ratioFre)
            ratioFre['Lg10LemmaFre'] = np.log10(sum(dist['wordforms']))  # combined lemma (stem) freq

            for group in [H, deltaH, ratioH, ratioFre]:
                for measure in group:
                    self._dict[record][measure] = group[measure]
                    if get_fieldnames:
                        if not measure in self.fieldnames:  # avoid duplicating fieldnames
                            self.fieldnames.append(measure)
            get_fieldnames = False

        if warn_msg is True:
            print "WARNING: not all items were found in SUBTLEX.\n" \
                  "Values for these items and other items of the same lemma were calculated using CELEX."

    def derivational_family_size(self):
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
            item = self._dict[record]['lemma_headword']
            decompositions = by_morpheme.get(item)
            if decompositions is not None:
                self._dict[record]['derivational_family_size'] = len(decompositions)
            else:
                self._dict[record]['derivational_family_size'] = 0
        if not 'derivational_family_size' in self.fieldnames:
            self.fieldnames.append('derivational_family_size')

    def derivational_family_entropy(self):
        by_morpheme = {}
        for lemma in clx._lemmas:
            if len(lemma['Parses']) < 1:
                continue
            decomposition = lemma['Parses'][0]['Imm']
            morphemes = decomposition.split('+')
            if morphemes[0] in self._dict.keys():
                by_morpheme.setdefault(morphemes[0], list()).append(lemma)

        for record in self._dict:
            item = self._dict[record]['lemma_headword']
            derived = by_morpheme.get(item)
            freqs = [x['Cob'] for x in derived]
            self._dict[record]['derivational_entropy'] = self.entropy(freqs)
        if not 'derivational_entropy' in self.fieldnames:
            self.fieldnames.append('derivational_entropy')

    def entropy(self, freq_vec, smoothing_constant=1):
        """This flat smoothing is an OK default but probably not the best idea:
        might be better to back off to the average distribution for the
        relevant paradigm (e.g. if singular forms are generally twice as likely
        as plural ones, it's better to use [2, 1] as the "prior" instead of
        [1, 1]).
        """
        freq_vec = list(freq_vec)  # change from set to list to add smoothing_constant
        vec = np.asarray(freq_vec, float) + smoothing_constant
        if sum(vec) == 0:
            return 0  # the event does not occur
        probs = vec / sum(vec)
        # Make sure we're not taking the log of 0 (by convention if p(x) = 0
        # then p(x) * log(p(x)) = 0 in the definition of entropy)
        probs[probs == 0] = 1
        return -np.sum(probs * np.log2(probs))

def debug():
    lex = LexVars(path_to_ds)

    # test all methods that use corpora with American spellings
    lex.wordnet_synsets()
    lex.synset_ratio()
    lex.age_of_acquisition()
    lex.levenshtein_distance()
    lex.subtitles()
    lex.subtlex_verb_ratio()

    # test all methods that use CELEX with British spelling
    brit_spell = [w['Word'] for w in clx._wordforms]
    stems = [lex[w]['Stem'] for w in lex]
    stems = change_spelling(stems, 'British', brit_spell)
    lex._append_column(stems, 'lemma_headword')  # b/c lemma_headword method gets some headwords wrong
    lex.inflectional_entropy()
    lex.derivational_family_size()
    lex.derivational_family_entropy()
    return lex

def test_corr(measure, lexvars_objects=None):
    """Tests correlations between entropy measures calculated using CELEX and SUBTLEX.

    This demonstrates how small some of the correlations are!
    """
    if lexvars_objects is None:  # create the LexVars objects
        lex = LexVars(path_to_ds)
        brit_spell = [w['Word'] for w in clx._wordforms]
        lex.change_spelling('British', brit_spell)  # change to brit spelling
        stems = [lex[w]['Stem'] for w in lex]
        stems = change_spelling(stems, 'British', brit_spell)
        lex._append_column(stems, 'lemma_headword')
        lexs = (lex.inflectional_entropy(use_subtlex=True), lex.inflectional_entropy(use_subtlex=False))
    else:  # use already instantiated LexVars objects
        lexs = lexvars_objects

    l1, l2 = lexs
    compare = [[l1[x][measure] for x in l1], [l2[y][measure] for y in l2]]
    return np.corrcoef(compare)

test_measures = ['H_separate_bare', 'H_collapsed_bare', 'H_no_bare', 'H_wordforms', 'H_affixed_wordforms',
                 'H_stem', 'H_stemS', 'H_stemEd', 'H_stemIng', 'H_stemAndstemS_pos', 'H_stemAndstemS_wordforms',
                 'H_verbs', 'H_nouns', 'H_adjectives', 'H_collapsed_NV', 'H_collapsed_pos',
                 'ratioFre_stem_lemma', 'ratioFre_stemS_lemma', 'ratioFre_stemEd_lemma', 'ratioFre_stemIng_lemma',
                 'ratioFre_stemS_affixes', 'ratioFre_stemEd_affixes', 'ratioFre_stemIng_affixes',
                 'ratioFre_Vlemma_lemma', 'ratioFre_Vstem_Vlemma', 'ratioFre_VstemS_Vlemma',
                 'ratioFre_Nlemma_lemma', 'ratioFre_Nstem_Nlemma', 'ratioFre_NstemS_Nlemma'
                 ]

def test_corrs(measures, lexvars_objects=None):
    return {measure: test_corr(measure, lexvars_objects) for measure in measures}