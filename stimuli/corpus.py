__author__ = 'Andrew Jamison'

from collections import Mapping, OrderedDict
import csv


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
        self.varcon_path = None  # this gets set the first time change_spelling is called

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
            raise ValueError("Length of column object must match number of entries in Corpus.")
        for entry, index in zip(self._dict, xrange(len(column))):
            self._dict[entry][name] = column[index]  # will replace a column with the same name
        if not name in self.fieldnames:  # avoid duplicating fieldnames
            self.fieldnames.append(name)

    def _remove_column(self, name):
        """Removes the column given by 'name'.
        """
        for entry in self._dict:
            del self._dict[entry][name]
        self.fieldnames.remove(name)

    def get_column(self, name):
        """Returns the column given by 'name'.
        """
        return [self._dict[entry][name] for entry in self._dict]

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

    def merge_columns(self, id_column, columns_to_merge, new_name, index=None, remove_old=False):
        """
        Ex: c.merge_columns('Flect_Type', [xxxxPastTense_xxxx, xxxxPastParticiple_xxxx], 0, 'xxxxPast_xxxx')
        """
        row_ids = self.get_column(id_column)
        # unique ids ordered from longest to shortest (avoids matching an id that is a substring of another id)
        remaining_ids = sorted(list(set(row_ids)), key=lambda x: len(x), reverse=True)
        new_column = [None]*self.__len__()
        for col_name in columns_to_merge:
            current_column = self.get_column(col_name)
            col_id = None
            for row_id in remaining_ids:
                if index:  # search for id at (the end of) a specific index in the list str.split('_')
                    if col_name.split('_')[index].endswith(row_id):
                        col_id = row_id
                        remaining_ids.remove(row_id)
                        break
                else:  # search for id at any place in string preceding '_' or at end of string
                    if row_id + '_' in col_name or col_name.endswith(row_id):
                        col_id = row_id
                        remaining_ids.remove(row_id)
                        break
            for i, row_id in enumerate(row_ids):
                if row_id == col_id:
                    new_column[i] = current_column[i]
            if remove_old:
                self._remove_column(col_name)
        self._append_column(new_column, new_name)

    def condense_by_contrasts(self, id_column, contrasts, replace_id_with, index=None, remove_old=False):
        """
        Ex: c.condense_by_contrasts('Flect_Type', [('xxxxPastTense_xxxx', 'xxxxPastParticiple_xxxx'),
                                   ('xxxxPastParticiple_xxxx, xxxxPresentParticiple_xxxx)], 0, ['Past', 'Participle'])
        """
        # unique ids ordered from longest to shortest (avoids matching an id that is a substring of another id)
        ids = sorted(list(set(self.get_column(id_column))), key=lambda x: len(x), reverse=True)
        # {'id1': ['xxxx<$>_xxxx', 'yyyy<$>_yyyy', ...], 'id2': ['xxxx<$>_xxxx', 'yyyy<$>_yyyy', ...], ...}
        colType_by_id = {}
        remaining_col_names = self.fieldnames
        # group columns by id for those with an id in ids
        for id in ids:
            matches = []
            for col_name in remaining_col_names:
                if index:  # search for id at (the end of) a specific index in the list str.split('_')
                    split_col_name = col_name.split('_')
                    if len(split_col_name) > index:
                        # found id: strip the col name of id (at index) and append the new name to matches
                        if split_col_name[index].endswith(id):
                            split_col_name[index] = split_col_name[index].replace(id, '<$>')
                            stripped_id = '_'.join(split_col_name)
                            matches.append(stripped_id)
                else:  # search for id directly preceding any occurrence of '_' in string or at end of string
                    split_col_name = [s.replace(id, '<$>') if s.endswith(id) else s for s in col_name.split('_')]
                    stripped_id = '_'.join(split_col_name)
                    # found id: strip the col name of all occurrences of id and append the new name to matches
                    if '<$>' in stripped_id:
                        matches.append(stripped_id)
            colType_by_id[id] = matches
            # remove previous matches to avoid double-matching an id that is a substring of another id
            remaining_col_names = set(remaining_col_names) - set(matches)

        contrasts, replace_id_with = list(contrasts), list(replace_id_with)  # coerce to lists
        # sort by smallest contrast size (necessary because supersets will replace subsets)
        contrasts, replace_id_with = (list(pair) for pair in zip(*sorted(zip(contrasts, replace_id_with),
                                                                         key=lambda pair: len(pair[0]))))
        columns_merged = set()
        for contrast, replacement in zip(contrasts, replace_id_with):
            # get columns common to all ids in a contrast
            common = set.intersection(*[set(colType_by_id[condition]) for condition in contrast])
            for column_type in common:
                columns_to_merge = []
                new_name = column_type.replace('<$>', replacement)
                for id in contrast:
                    has_id = column_type.replace('<$>', id)
                    columns_to_merge.append(has_id)
                    columns_merged.add(has_id)
                # note: do not pass remove_old to this method; will break if the same column is in multiple contrasts
                # this gets handled separately below
                self.merge_columns(id_column, columns_to_merge, new_name, index, False)
        if remove_old:
            for old_col in columns_merged:
                self._remove_column(old_col)

    def _warning_msg(self, func_name, comparison, show_missing_items=False):
        if show_missing_items:
            print "WARNING: the following items from %s were not found and were given a value of 'None':\n %s" \
                % (func_name, self.compare_items(comparison, True)['not_in_comparison'])
        else:
            print "WARNING: at least one item from %s was not found and was given a value of 'None'!\n" \
                  "Set 'verbose' to 'True' to see the list of items." % func_name

    def change_spelling(self, change_to, compare_to=None, varcon_path=None):
        """Modifies the spelling of keys to American or British English.

        Optional second argument constrains spelling conversions in the following way:
        a key is changed only if the old spelling is not in the comparison object and the new
        spelling is in the comparison object. This is useful if you want to minimally change
        keys to match elements in the comparison object.

        Note: it is possible to recover the original spelling of each key from its <item_name>
        value supplied in __init__.
        """
        from brit_spelling import get_translations
        if self.varcon_path is None:  # this gets set the first time change_spelling is called
            if varcon_path is None:
                raise IOError('Could not find path to varcon.txt.')
            else:
                self.varcon_path = varcon_path
        translations = get_translations(self.varcon_path)  # must be able to find this directory
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

