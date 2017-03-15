import numpy as np
import scipy.io  as sio

#thanks to http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries

def loadmat(filename):
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dictobj):
    for key in dictobj:
        if isinstance(dictobj[key], sio.matlab.mio5_params.mat_struct):
            dictobj[key] = _todict(dictobj[key])
    return dictobj        

def _todict(matobj):
    dictobj = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dictobj[strg] = _todict(elem)
        #elif isinstance(elem,np.ndarray):
        #    dict[strg] = _tolist(elem)
        else:
            dictobj[strg] = elem
    return dictobj

def _tolist(ndarray):
    elem_list = []            
    for sub_elem in ndarray:
        if isinstance(sub_elem, sio.matlab.mio5_params.mat_struct):
            elem_list.append(_todict(sub_elem))
        elif isinstance(sub_elem,np.ndarray):
            elem_list.append(sub_elem)
        else:
            elem_list.append(sub_elem)
    return elem_list