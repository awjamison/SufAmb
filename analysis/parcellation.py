'''
Created on Aug 17, 2014

@author: andrew
'''

from os import path, environ
from mne import read_labels_from_annot
from eelbrain.mne_fixes import write_labels_to_annot

f = path.join('/Volumes', 'Backup', 'sufAmb', 'mri')
environ["SUBJECTS_DIR"] = f  # set env var so mne can find subjects dir

def make_parc(e):
    """Creates a parcellation of the left frontal, parietal, and temporal lobes.
    """
    lab = read_labels_from_annot('fsaverage', 'PALS_B12_Lobes', 'lh')
    # use the modified write_labels_to_annot function from eelbrain
    write_labels_to_annot(labels=[lab[1], lab[4], lab[5]], subject='fsaverage', parc='frontal_temporal_parietal',
                          overwrite=True)
    return [lab[1], lab[4], lab[5]]

# remove files: e.rm('file-evoked')
