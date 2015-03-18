'''
Created on Aug 17, 2014

@author: andrew
'''

import os
from mne import read_labels_from_annot
from eelbrain.mne_fixes import write_labels_to_annot

# set env var so mne can find subjects dir
os.environ["SUBJECTS_DIR"] = os.path.join('/Volumes', 'Backup', 'sufAmb', 'mri')

def make_parcellation(parc_name, from_annot, labels, hemi='both'):
    all_labs = read_labels_from_annot('fsaverage', from_annot, hemi)
    new_labs = [l for l in all_labs if l.name.split('-')[0] in labels]
    write_labels_to_annot(labels=new_labs, subject='fsaverage', parc=parc_name, overwrite=True)

def make_sufamb_parcellations():
    """Parcellations for SufAmb experiment
    """

    BAs = []  # brodmann areas
    aparcs = []  # anatomical DKT atlas (Desikan et al., 2006)

    # angular gyrus
    BAs.append('39'), aparcs.append('inferiorparietal')
    # middle temporal gyrus
    BAs.append('21'), aparcs.append('middletemporal')
    # temporal pole
    BAs.append('38'), aparcs.append('temporalpole')
    # inferior frontal gyrus (pars orbitalis, triangularis, and opercularis)
    BAs.extend(['44', '45', '47']), aparcs.extend(['parsorbitalis', 'parstriangularis', 'parsopercularis'])
    # ventromedial PFC / medial orbitofrontal
    BAs.extend(['25', '12', '11', '32', '10']), aparcs.append('medialorbitofrontal')
    # dorsomedial PFC
    BAs.extend(['8', '9']), aparcs.append('superiorfrontal')
    # isthmus cingulate (sometimes "posterior cingulate")
    BAs.extend(['31', '23', '30', '29', '26']), aparcs.append('isthmuscingulate')

    BAs = ['Brodmann.' + ba for ba in BAs]  # 'Brodmann' prefix

    # make left hemi parcellations
    make_parcellation('SufAmb_aparc', 'aparc', aparcs, 'lh')
    make_parcellation('SufAmb_Brodmann', 'PALS_B12_Brodmann', BAs, 'lh')
