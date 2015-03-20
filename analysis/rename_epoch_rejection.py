__author__ = 'andrew'

import os
from eelbrain import load

def rename(old, new):
    """Renames epoch rejection files.
    """
    meg_path = os.path.join('/Volumes', 'Backup', 'sufAmb', 'meg')
    subjects = os.listdir(meg_path)
    if '.DS_Store' in subjects:
        subjects.remove('.DS_Store')
    for s in subjects:
        rejection_path = os.path.join(meg_path, s, 'epoch selection')
        if os.path.exists(os.path.join(rejection_path, old)):
            rejection_file = load.unpickle(os.path.join(rejection_path, old))
            rejection_file.save_pickled(os.path.join(rejection_path, new))

if __name__ == '__main__':
    rename('sufAmb_1-40_epoch-man.pickled', 'sufAmb_1-40_100-600-man.pickled')
    rename('sufAmb_1-40_epoch450-man.pickled', 'sufAmb_1-40_150-450-man.pickled')
    print 'Done'
