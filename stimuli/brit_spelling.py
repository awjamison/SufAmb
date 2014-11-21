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

