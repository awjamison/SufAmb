__author__ = 'Andrew Jamison'
# VarCon (Variant Conversion Info) home: http://wordlist.aspell.net/varcon/
# sourceforge: http://sourceforge.net/projects/wordlist/files/VarCon/2014.08.11/

from os.path import expanduser, join

home = expanduser("~")
corPath = join(home, "Dropbox", "corpora")

def get_translations(cor_path=corPath):
    """ Reads and parses the VarCon text file.\n
    Returns list of dictionaries containing Brit and American spellings.
    """
    translations = []
    varconPath = join(cor_path, "varcon.txt")
    with open(varconPath) as f:
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

def change_spelling(words, change_to, compare_to=None, cor_path=corPath):
    """Modifies the spelling of elements in a list 'words' to American or British English.\n
    Optional third argument constrains spelling conversions in the following way:\n
    an element is changed only if the old spelling is not in the comparison list and the new\n
    spelling is in the comparison list. This is useful if you want to minimally change\n
    elements to match those in the comparison list.
    """
    translations = get_translations(cor_path)  # must be able to find this directory
    new_sp = {x[change_to] for x in translations}
    if change_to == 'American':
        change_sp = {x['British']: x['American'] for x in translations}
    else:
        change_sp = {x['American']: x['British'] for x in translations}
    if compare_to is not None:
        # changes new_sp to union of new_sp and compare_to
        # changes change_sp to intersection of change_sp and compare_to
        in_comparison = set(compare_to)
        new_sp.update(in_comparison)
        change_sp = {k: change_sp[k] for k in change_sp if change_sp[k] in in_comparison}

    renamed = []
    for entry in words:
        if entry in new_sp:
            renamed.append(entry)
            continue  # avoid replacing a word when its spelling matches an item in the set of new spellings.
        if entry in change_sp:
            spelling = change_sp[entry]
            renamed.append(spelling)
    return renamed
