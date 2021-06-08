import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import pandas as pd
from colors import *
from mutators import addOneLayer
import os, subprocess
from globalInfos import type123_heads_tails_extraction,\
                        config_extraction,\
                        extra_info_extraction
from globalInfos import MODELNAMES,\
                        MOD,\
                        ORIGIN_PATH,\
                        MUTANT_PATH,\
                        ORDERS,\
                        OPSPOOL,\
                        TOTALNUMBER,\
                        EACHNUMBER

def run_random_mod():
    for order in ORDERS:
        for cnt in range(1, upperbound):
            if order == 1:
                newmodel = addOneLayer(model)
                dest = os.path.join(MUTANT_PATH, str(order))
                if not os.path.exists(dest):
                    subprocess.check_output(f'mkdir -p {dest}', shell=True)
                h5dest = os.path.join(dest, f'lenet5-mnist_{str(cnt)}.h5')
                newmodel.save(h5dest)
            else:
                order1_dest = os.path.join(MUTANT_PATH, str(order-1))
                h5files = os.listdir(order1_dest)
                # htfile = np.random.choice(h5files)
                htfile = ''
                for file in h5files:
                    if file.endswith(f'_{cnt}.h5'):
                        htfile = file
                        break
                if not htfile:
                    raise Exception('Find no h5file')
                htfile_name = htfile[:-3]
                htfile_newname = htfile_name + '_' + str(cnt) + '.h5'

                newmodel = addOneLayer(model)
                dest = os.path.join(MUTANT_PATH, str(order))
                if not os.path.exists(dest):
                    subprocess.check_output(f'mkdir -p {dest}', shell=True)
                h5dest = os.path.join(dest, htfile_newname)
                newmodel.save(h5dest)

def run_fixed_mod():
    for order in ORDERS:
        for op in OPSPOOL:
            for cnt in range(1, upperbound):
                if order == 1:
                    newmodel = addOneLayer(model, mode = 'fixed', op = op)
                    dest = os.path.join(MUTANT_PATH, str(order))
                    if not os.path.exists(dest):
                        subprocess.check_output(f'mkdir -p {dest}', shell=True)
                    h5dest = os.path.join(dest, f'lenet5-mnist_{str(cnt)}.h5')
                    newmodel.save(h5dest)
                else:
                    order1_dest = os.path.join(MUTANT_PATH, str(order-1))
                    h5files = os.listdir(order1_dest)
                    # htfile = np.random.choice(h5files)
                    htfile = ''
                    for file in h5files:
                        if file.endswith(f'_{cnt}.h5'):
                            htfile = file
                            break
                    if not htfile:
                        raise Exception('Find no h5file')
                    htfile_name = htfile[:-3]
                    htfile_newname = htfile_name + '_' + str(cnt) + '.h5'

                    newmodel = addOneLayer(model)
                    dest = os.path.join(MUTANT_PATH, str(order))
                    if not os.path.exists(dest):
                        subprocess.check_output(f'mkdir -p {dest}', shell=True)
                    h5dest = os.path.join(dest, htfile_newname)
                    newmodel.save(h5dest)

if __name__ == '__main__':

    config_extraction()
    extra_info_extraction()

    for modelname in MODELNAMES:
        
        modelpath = os.path.join(ORIGIN_PATH, f'{modelname}_origin.h5')
        model = keras.models.load_model(modelpath)
        type123_heads_tails_extraction(model.layers, len(model.layers))
        if MOD == 'random':
            upperbound = TOTALNUMBER + 1
            run_random_mod()
        elif MOD == 'fixed':
            upperbound = EACHNUMBER + 1
            run_fixed_mod()
        else:
            raise Exception(Cyan(f'Unkown mode: {MODE}'))
