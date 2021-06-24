import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import pandas as pd
from colors import *
from mutators import addOneLayer
import os, subprocess
from globalInfos import edge_collection,\
                        config_extraction,\
                        extra_info_extraction,\
                        available_edges_extraction_for4types

def _test_mutant(mutant_path):
    model = keras.models.load_model(mutant_path)
    # model.summary()
    print(Magenta('TEST SUCCEED!'))

def run_random_mod(model, modelname):
    for order in ORDERS:
        for cnt in range(1, upperbound):
            if int(order) == 1:
                newmodel = addOneLayer(model)
                dest = os.path.join(os.path.join(MUTANT_PATH, modelname), str(order))
                if not os.path.exists(dest):
                    subprocess.check_output(f'mkdir -p {dest}', shell=True)
                h5dest = os.path.join(dest, f'{modelname}_{str(cnt)}.h5')
                newmodel.save(h5dest)
            else:
                order1_dest = os.path.join(os.path.join(MUTANT_PATH, modelname), str(int(order)-1))
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
                dest = os.path.join(os.path.join(MUTANT_PATH, modelname), str(order))
                if not os.path.exists(dest):
                    subprocess.check_output(f'mkdir -p {dest}', shell=True)
                h5dest = os.path.join(dest, htfile_newname)
                newmodel.save(h5dest)

def run_fixed_mod(model, modelname):
    for order in ORDERS:
        for op in OPSPOOL:
            print(Yellow(op))
            # model.summary()
            for cnt in range(1, upperbound):
                if int(order) == 1:
                    newmodel = addOneLayer(model, mode = 'fixed', op = op)
                    dest = os.path.join(os.path.join(MUTANT_PATH, modelname), str(order))
                    if not os.path.exists(dest):
                        subprocess.check_output(f'mkdir -p {dest}', shell=True)
                    
                    h5dest = os.path.join(dest, f'{modelname}_{op}_{str(cnt)}.h5')
                    newmodel.save(h5dest)
                else:
                    order1_dest = os.path.join(os.path.join(MUTANT_PATH, modelname), str(int(order)-1))
                    if not os.path.exists(order1_dest):
                        subprocess.check_output(f'mkdir -p {order1_dest}', shell=True)
                    h5files = os.listdir(order1_dest)
                    htfile = np.random.choice(h5files)
                    # htfile = ''
                    # for file in h5files:
                    #     if file.endswith(f'_{cnt}.h5'):
                    #         htfile = file
                    #         break
                    # if not htfile:
                    #     raise Exception('Find no h5file')
                    htfile_name = htfile[:-3]
                    htfile_newname = htfile_name + '_' + op + '_' + str(cnt) + '.h5'

                    newmodel = addOneLayer(model)
                    dest = os.path.join(os.path.join(MUTANT_PATH, modelname), str(order))
                    if not os.path.exists(dest):
                        subprocess.check_output(f'mkdir -p {dest}', shell=True)
                    h5dest = os.path.join(dest, htfile_newname)
                    newmodel.save(h5dest)
                # REMOVED
                _test_mutant(h5dest)

if __name__ == '__main__':

    config_extraction()
    extra_info_extraction()
    from globalInfos import MODELNAMES,\
                            MODE,\
                            ORIGIN_PATH,\
                            MUTANT_PATH,\
                            ORDERS,\
                            OPSPOOL,\
                            TOTALNUMBER,\
                            EACHNUMBER
    for modelname in MODELNAMES:
        print(Green(modelname))
        modelpath = os.path.join(ORIGIN_PATH, f'{modelname}_origin.h5')
        model = keras.models.load_model(modelpath)
        # model.summary()
        edge_collection(model)
        available_edges_extraction_for4types()
        if MODE == 'random':
            upperbound = int(TOTALNUMBER) + 1
            run_random_mod(model, modelname)
        elif MODE == 'fixed':
            upperbound = int(EACHNUMBER) + 1
            run_fixed_mod(model, modelname)
        else:
            raise Exception(Cyan(f'Unkown mode: {MODE}'))
