__author__ = 'Andrew Jamison'

from os.path import expanduser, join
import csv

home = expanduser("~")
expPath = join(home, "Dropbox", "all_subjects.csv")  # path to custom list; edit
corPath = join(home, "Dropbox", "corpora")  # path to corpora; edit
elpPath = join(corPath, "ELP", "ELPfull.csv")
clxPath = join(corPath, "CELEX_V2")
slxPath = join(corPath, "SUBTLEX-US.csv")
aoaPath = join(corPath, "AoA.csv")
dbParams = [(elpPath, 'elp'), (slxPath, 'slx'), (aoaPath, 'aoa')]


class Corpus(object):

    def __init__(self):
        self.db = {}
        self.dbinfo = {}
    
    def read(self, path, db_name, item_name='Word'):
        with open(path) as f:
            reader = csv.DictReader(f)
            self.db[db_name] = list(reader)
            self.dbinfo[db_name] = {'item_name': item_name}

    def read_multiple(self, dbs=dbParams, include_clx=True, clx_path=clxPath):
        """Reads multiple databases. Takes a list list of tuples (path, db_name)
        and passes them as arguments to 'read'. Can provide optional third
        element item_name, otherwise item_name=='Word'.
        """
        for db in dbs:
            if len(db) == 2:
                self.read(db[0], db[1])
            elif len(db) == 3:
                self.read(db[0], db[1], db[2])
            else:
                raise ValueError("Incorrect number of elements.")
        if include_clx:
            self.read_clx(clx_path)

    def read_clx(self, clx_path=clxPath):
        """Special function for reading and combining sections of the CELEX database.
        Make sure csv files are alphabetized (unchanged) before calling this.
        """
        groups = {'lemmas': ['efl', 'eml', 'eol', 'epl', 'esl'],
                  'wordforms': ['efw', 'emw', 'eow', 'epw'],
                  'syllables': ['efs'],
                  'types': ['etc']}
        item_names = {'lemmas': 'Head', 'wordforms': 'Word', 'syllables': 'Syllable', 'types': 'Type'}
        for group in groups:
            path = join(clx_path, groups[group][0]+".csv")
            with open(path) as f:
                combined = list(csv.DictReader(f))  # first db
            if len(group) > 1:
                for db in groups[group][1:]:
                    path = join(clx_path, db+".csv")
                    with open(path) as f:
                        l = list(csv.DictReader(f))
                    i = 0
                    for x in l:  # add new entries
                        combined[i].update(x)
                        i += 1
            db_name = 'clx-'+group
            self.db[db_name] = combined
            self.dbinfo[db_name] = {'itemName': item_names[group]}

    def check_matches(self, db1, db2, show_items=False):
        """Compares items from two databases.
        """
        col1 = self.dbinfo[db1]['itemName']
        col2 = self.dbinfo[db2]['itemName']
        col1 = {x[col1] for x in self.db[db1]}
        col2 = {x[col2] for x in self.db[db2]}
        if show_items:
            notindb1 = col1 - col2
            notindb2 = col2 - col1
            return notindb1, notindb2
        else:
            if col1 == col2: return True, True
            elif col1 <= col2: return True, False
            elif col2 <= col1: return False, True
            else: return False, False