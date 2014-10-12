'''
Created on Aug 17, 2014

@author: andrew
'''

from mne import read_labels_from_annot
from eelbrain.mne_fixes import write_labels_to_annot #@UnresolvedImport

def make_parc(e):
    lab = read_labels_from_annot('fsaverage', 'custom_par', 'lh')
    lastparc = lab[len(lab)-1].name
    if lastparc.__contains__('unknown') == True: del lab[len(lab)-1] # remove 'unknown' (=rest of brain)
    frontal = [lab[0],lab[1],lab[3],lab[4],lab[6],lab[7]]
    temporal = [x for x in lab if x not in frontal]
    print '-----------make sure these are frontal:-----------'
    for f in frontal:
        print f.name
    print '-----------make sure these are temporal:-----------'
    for t in temporal:
        print t.name
    
    combinedFrontal = frontal[0]
    for i in range(1,len(frontal)):
        combinedFrontal = combinedFrontal + frontal[i]
    combinedFrontal.name = 'frontal'
        
    combinedTemporal = temporal[0]
    for i in range(1,len(temporal)):
        combinedTemporal = combinedTemporal + temporal[i]
    combinedTemporal.name = 'temporal'
    
    # use the modified write_labels_to_annot function from eelbrain
    write_labels_to_annot(labels=[combinedFrontal, combinedTemporal], subject='fsaverage', parc='frontal_temporal', overwrite=True)
    return [combinedFrontal, combinedTemporal]

# custom parc:
# labs = mne.read_labels_from_annot('fsaverage')
# custom_par_i = [4,10,12,24,28,30,54,56,60,64,66]
# custom_par = [labs[i] for i in custom_par_i]
# mne.write_labels_to_annot(labels=custom_par, parc='custom_par')
# mne.write_labels_to_annot(labels=custom_par, subject='fsaverage', parc='custom_par', subjects_dir=sd)
# remove files: e.rm('file-evoked')
