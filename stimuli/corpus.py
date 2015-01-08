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
varconPath = path.join(corPath, "varcon.txt")
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

    def _append_column(self, column, name):
        """Column is a list-like object with len(column) == len(corpus).
        """
        column = tuple(column)
        if len(column) != self.__len__():
            raise ValueError("Length of column object must match length of Corpus instance.")
        for entry, index in zip(self._dict, xrange(len(column))):
            self._dict[entry][name] = column[index]  # will replace a column with the same name
        if not name in self.fieldnames:  # avoid duplicating fieldnames
            self.fieldnames.append(name)

    def compare_items(self, compare_to, show_missing_items=False):
        """Compares keys to the keys of another Corpus instance or to elements of a list-like object.
        """
        if isinstance(compare_to, Corpus):
            in_comparison = set(compare_to.keys())
        else:
            in_comparison = set(compare_to)
        if show_missing_items:
            in_corpus = set(self._dict.keys())
            not_in_corpus = in_comparison - in_corpus
            not_in_comparison = in_corpus - in_comparison
            return {'not_in_corpus': not_in_corpus, 'not_in_comparison': not_in_comparison}
        else:
            in_corpus = []
            for entry in self._dict:
                if entry in in_comparison:
                    in_corpus.append(True)
                else:
                    in_corpus.append(False)
            return in_corpus

    def has_value(self, column, expression, value, show_missing_values=False):
        """For each item in Corpus class, True if <column expression value> evaluates to True.\n
        Optionally returns the set of items for which the expression evaluates to False.
        """
        if type(value) is str:
            re = "self._dict[entry]['%s'] %s '%s'" % (column, expression, value)
        else:
            re = "self._dict[entry]['%s'] %s %s" % (column, expression, value)
        if show_missing_values:
            return {entry for entry in self._dict if eval(re) is False}
        else:
            return [eval(re) for entry in self._dict]

    def copy_columns(self, copy_from, columns, show_missing_items=False):
        has_item = self.compare_items(copy_from)
        if isinstance(columns, str):  # single column
            columns = [columns]
        else:  # vector of columns
            columns = list(columns)
        if False in has_item:  # print warning message if item not found in 'copy_from'
            if isinstance(copy_from, Corpus):
                self._warning_msg(copy_from.name, copy_from, show_missing_items)
            else:
                self._warning_msg(str(copy_from), copy_from, show_missing_items)
        new_values = []
        for entry, exists in zip(self._dict, has_item):  # iterate over each item
            if exists:
                item_values = []
                for i in xrange(len(columns)):  # iterate over each variable
                    value = copy_from[entry][columns[i]]
                    item_values.append(value)
            else:
                item_values = [None]*len(columns)
            new_values.append(item_values)
        new_columns = zip(*new_values)  # transpose to list of columns
        for column, i in zip(new_columns, xrange(len(columns))):
            self._append_column(column, columns[i])

    def _warning_msg(self, func_name, comparison, show_missing_items=False):
        if show_missing_items:
            print "WARNING: the following items from %s were not found and were given a value of 'None':\n %s" \
                % (func_name, self.compare_items(comparison, True)['not_in_comparison'])
        else:
            print "WARNING: at least one item from %s was not found and was given a value of 'None'!\n" \
                  "Set 'verbose' to 'True' to see the list of items." % func_name

    def change_spelling(self, change_to, compare_to=None):
        """Modifies the spelling of keys to American or British English.

        Optional second argument constrains spelling conversions in the following way:
        a key is changed only if the old spelling is not in the comparison object and the new
        spelling is in the comparison object. This is useful if you want to minimally change
        keys to match elements in the comparison object.

        Note: it is possible to recover the original spelling of each key from its <item_name>
        value supplied in __init__.
        """
        from brit_spelling import get_translations
        translations = get_translations(varconPath)  # must be able to find this directory
        new_sp = {x[change_to] for x in translations}  # creates a set of spellings in the chosen dialect
        if change_to == 'American':
            change_sp = {x['British']: x['American'] for x in translations}  # British to American
        elif change_to == 'British':
            change_sp = {x['American']: x['British'] for x in translations}  # American to British
        else:
            raise ValueError("Unsupported dialect.")
        if compare_to is not None:
            # changes new_sp to union of new_sp and compare_to
            # changes change_sp to intersection of change_sp and compare_to
            if isinstance(compare_to, Corpus):
                in_comparison = set(compare_to.keys())
            else:
                in_comparison = set(compare_to)
            new_sp.update(in_comparison)
            change_sp = {k: change_sp[k] for k in change_sp if change_sp[k] in in_comparison}

        renamed = OrderedDict()  # new keys are created and values are copied from self._dict
        for entry in self._dict:
            if entry in new_sp:  # don't change the spelling; it already matches an element in the new dialect
                renamed[entry] = self._dict[entry]
            elif entry in change_sp:  # the spelling does not exist in the original dialect but a translation exists
                spelling = change_sp[entry]
                renamed[spelling] = self._dict[entry]
            else:  # the word could not be found
                renamed[entry] = self._dict[entry]
        self._dict = renamed

    def write(self, path):
        """Writes data to a csv file.
        """
        with open(path, 'w') as f:
            writer = csv.DictWriter(f, delimiter=',', fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self._dict.values())

