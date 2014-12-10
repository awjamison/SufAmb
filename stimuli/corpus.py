__author__ = 'Andrew Jamison'

from collections import Mapping, OrderedDict
from os import path
import csv

# some helpful directories
home = path.expanduser("~")
expPath = path.join(home, "Dropbox", "SufAmb", "master_list35_update3.csv")  # path to custom list; edit
corPath = path.join(home, "Dropbox", "corpora")  # path to corpora; edit
elpPath = path.join(corPath, "ELP", "ELPfull.csv")
clxPath = path.join(corPath, "CELEX_V2")
slxPath = path.join(corPath, "SUBTLEX-US.csv")
aoaPath = path.join(corPath, "AoA.csv")
outPath = path.join(home, "Dropbox", "SufAmb", "this_is_a_test.csv")

class Corpus(Mapping):
    """Special data container for a table of words and their properties.\n
    Reads a csv file and constructs an ordered dictionary with additional attributes.
    """

    def __init__(self, path_to_file, item_name='Word'):
        self.name = path_to_file  # shows file origin as default
        with open(path_to_file, 'r') as f:
            reader = csv.DictReader(f)
            self.fieldnames = reader.fieldnames
            db = list(reader)
        self._dict = OrderedDict((entry[item_name], entry) for entry in db)
        if self.__len__() != len(db):
            print "WARNING: duplicate entries for '%s'. Some entries were overwritten." % item_name

    def __getitem__(self, key):
        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def compare_items(self, compare_to, show_items=False):
        """Compares keys to the keys of another Corpus instance or to elements of a list-like object.
        """
        in_corpus = set(self._dict.keys())
        if isinstance(compare_to, Corpus):
            in_comparison = set(compare_to.keys())
        else:
            in_comparison = set(compare_to)
        if show_items:
            not_in_corpus = in_corpus - in_comparison
            not_in_comparison = in_comparison - in_corpus
            return {'not_in_corpus': not_in_corpus, 'not_in_comparison': not_in_comparison}
        else:
            if in_corpus == in_comparison: return True, True
            elif in_corpus <= in_comparison: return True, False
            elif in_comparison <= in_corpus: return False, True
            else: return False, False

    def change_spelling(self, change_to, compare_to=None):
        """Modifies the spelling of keys to American or British English.\n
        Optional second argument constrains spelling conversions in the following way:\n
        a key is changed only if the old spelling is not in the comparison object and the new\n
        spelling is in the comparison object. This is useful if you want to minimally change\n
        keys to match elements in the comparison object.\n
        Note: it is possible to recover the original spelling of each key from its <item_name>\n
        value supplied in __init__.
        """
        from brit_spelling import get_translations
        translations = get_translations(corPath)  # must be able to find this directory
        new_sp = {x[change_to] for x in translations}
        if change_to == 'American':
            change_sp = {x['British']: x['American'] for x in translations}
        else:
            change_sp = {x['American']: x['British'] for x in translations}
        if compare_to is not None:
            # changes new_sp to union of new_sp and compare_to
            # changes change_sp to intersection of change_sp and compare_to
            if isinstance(compare_to, Corpus):
                in_comparison = set(compare_to.keys())
            else:
                in_comparison = set(compare_to)
            new_sp.update(in_comparison)
            change_sp = {k: change_sp[k] for k in change_sp if change_sp[k] in in_comparison}

        renamed = OrderedDict()
        for entry in self._dict:
            if entry in new_sp:
                renamed[entry] = self._dict[entry]
                continue  # avoid replacing a word when its spelling matches an item in the set of new spellings.
            if entry in change_sp:
                spelling = change_sp[entry]
                renamed[spelling] = self._dict[entry]
        self._dict = renamed

    def write(self, path):
        """Writes data to a csv file.
        """
        with open(path, 'w') as f:
            writer = csv.DictWriter(f, delimiter=',', fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self._dict.values())

