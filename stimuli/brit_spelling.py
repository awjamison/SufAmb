__author__ = 'Andrew Jamison'
# VarCon (Variant Conversion Info) home: http://wordlist.aspell.net/varcon/
# sourceforge: http://sourceforge.net/projects/wordlist/files/VarCon/2014.08.11/

from os.path import expanduser, join

home = expanduser("~")
corPath = join(home, "Dropbox", "corpora")
varconPath = join(corPath, "varcon.txt")

def get_translations(varcon_path=varconPath):
    """Creates a list of spelling translations from the VarCon text file.

    Reads and parses the VarCon text file (http://wordlist.aspell.net/varcon/).
    Individual entries (words) in VarCon are converted into dictionaries that
    contain the American and British spelling variants. A word with multiple
    spellings in a dialect will occur more than once in the list and may
    map the same value.
    """
    translations = []
    with open(varcon_path) as f:
        reader = f.readlines()
        for line in reader:
            if not line.startswith("#"):
                if not line.find("/") == -1:
                    line = line.replace("\n", "")
                    spellings = line.split("/")
                    american = spellings[0].split(":")[1]
                    american = american.strip()
                    brit = spellings[1].split(":")[1]
                    brit = brit.strip()
                    translations.append({'American': american, 'British': brit})
    return translations

def change_spelling(words, change_to, compare_to=None, varcon_path=varconPath):
    """Modifies the spelling of elements in a list 'words' to American or British English.

    Optional third argument constrains spelling conversions in the following way:
    an element is changed only if the old spelling is not in the comparison list and the new
    spelling is in the comparison list. This is useful if you want to minimally change
    elements to match those in the comparison list.
    """
    translations = get_translations(varcon_path)  # must be able to find this directory
    new_sp = {x[change_to] for x in translations}  # creates a set of spellings in the chosen dialect
    if change_to == ('American' or 'american'):
        change_sp = {x['British']: x['American'] for x in translations}  # British to American
    elif change_to == ('British' or 'british'):
        change_sp = {x['American']: x['British'] for x in translations}  # American to British
    else:
        raise ValueError("Unsupported dialect.")
    if compare_to is not None:
        # changes new_sp to union of new_sp and compare_to
        # changes change_sp to intersection of change_sp and compare_to
        in_comparison = set(compare_to)
        new_sp.update(in_comparison)
        change_sp = {k: change_sp[k] for k in change_sp if change_sp[k] in in_comparison}

    renamed = []
    for entry in words:
        if entry in new_sp:  # don't change the spelling; it already matches an element in the new dialect
            renamed.append(entry)
        elif entry in change_sp:  # the spelling does not exist in the original dialect but a translation exists
            spelling = change_sp[entry]
            renamed.append(spelling)
        else:   # the word could not be found
            renamed.append(entry)
    return renamed
