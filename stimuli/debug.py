__author__ = 'Andrew Jamison'

import os
from corpus import Corpus
from lex_dec_vars import *

# some helpful directories
home = os.path.expanduser("~")
expPath = os.path.join(home, "Dropbox", "SufAmb", "master_list35_update3_words.csv")  # path to custom list; edit
corPath = os.path.join(home, "Dropbox", "corpora")  # path to corpora; edit
elpPath = os.path.join(corPath, "ELP", "ELPfull.csv")
clxPath = os.path.join(corPath, "CELEX_V2")
slxPath = os.path.join(corPath, "SUBTLEX-US.csv")
aoaPath = os.path.join(corPath, "AoA.csv")
varconPath = os.path.join(corPath, "varcon.txt")
outPath = os.path.join(home, "Dropbox", "SufAmb", "this_is_a_test.csv")


def test_all():
    lex = LexVars(expPath)

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
    lex.change_spelling('British', brit_spell, varconPath)
    lex.inflectional_information(use_subtlex=False)
    lex.derivational_family_size()
    lex.derivational_family_entropy()
    return lex

def test_corr(measure, lexvars_objects=None):
    """Tests correlations between entropy measures calculated using CELEX and SUBTLEX.
    This demonstrates how small some of the correlations are!
    """
    if lexvars_objects is None:  # create the LexVars objects
        lex = LexVars(expPath)
        brit_spell = [w['Word'] for w in clx._wordforms]
        lex.change_spelling('British', brit_spell)  # change to brit spelling
        stems = [lex[w]['Stem'] for w in lex]
        stems = change_spelling(stems, 'British', brit_spell)
        lex._append_column(stems, 'lemma_headword')
        lexs = (lex.inflectional_information(use_subtlex=True), lex.inflectional_information(use_subtlex=False))
    else:  # use already instantiated LexVars objects
        lexs = lexvars_objects

    l1, l2 = lexs
    compare = [[l1[x][measure] for x in l1], [l2[y][measure] for y in l2]]
    return np.corrcoef(compare)

test_measures = ['H_wordforms', 'H_affixed_wordforms',
                 'H_stem', 'H_stemS', 'H_stemEd', 'H_stemIng', 'H_stemAndstemS',
                 'H_verbs', 'H_nouns', 'H_adjectives',
                 ]

def test_corrs(measures, lexvars_objects=None):
    return {measure: test_corr(measure, lexvars_objects) for measure in measures}

def test_condense():
    c = Corpus(expPath)
    c._append_column(['c'+str(j) for i in range(len(c)/5) for j in range(1,6)], 'id_col')  # create five conditions
    # create a variable shared by three conditions
    c._append_column([1]*len(c), 'xxx_xxxc1_xxx')
    c._append_column([2]*len(c), 'xxx_xxxc2_xxx')
    c._append_column([3]*len(c), 'xxx_xxxc3_xxx')
    # create another variable shared by three conditions, two of which share the previous variable
    c._append_column([2]*len(c), 'yyy_yyyc2_yyy')
    c._append_column([3]*len(c), 'yyy_yyyc3_yyy')
    c._append_column([4]*len(c), 'yyy_yyyc4_yyy')

    contrasts = [('c1', 'c2', 'c3'), ('c2', 'c3', 'c4')]
    replace_id_with = ['C1C2C3', 'C2C3C4']
    c.condense_by_contrasts('id_col', contrasts, 1, replace_id_with, remove_old=True)

    c.write(outPath)  # easiest just to check by viewing csv file
    return c