#**********************************************#
# The preproccessing of quarternary structures.#
#   (creating pkl out of preprocced pkls)      #
#**********************************************#

import os
# import gzip as gz
import numpy as np
# import Bio as bio
import pickle as pkl
# import Bio.PDB as pdb
import cosma_algebra as ca

def gen3_read_pkl(file) -> list[dict]:
    """Trying to extract dimer from #pdb# PKL file."""

    with open(file, 'rb') as file:
        model_list = pkl.load(file)

    if len(model_list) < 2:
        raise Exception('Not enough chains')

    id = model_list[0]['id']
    idxs = {model['idx'] for model in model_list}

    dimers = []
    singles = []

    for idx in idxs:
        models = [model for model in model_list if model['idx'] == idx]
        if len(models) == 2:
            if ca.check_default(models[0]['coord'], models[1]['coord']):
                dimers.append({'id': id, 'homo': ca.check_homo(models[0]['amino'], models[1]['amino']), 'singles': False,
                               'crop':  (models[0]['crop'],  models[1]['crop']),
                               'amino': (models[0]['amino'], models[1]['amino']),
                               'coord': (models[0]['coord'], models[1]['coord']),
                               'pmtrx': (models[0]['pmtrx'], models[1]['pmtrx'])})
        elif len(models) == 1:
            singles.append(models[0])
    if len(dimers) == 0:
        if len(singles) == 2 and ca.check_default(singles[0]['coord'], singles[1]['coord']):
            return id, [{'id': id, 'homo': ca.check_homo(singles[0]['amino'], singles[1]['amino']), 'singles': True,
                     'crop':  (singles[0]['crop'],  singles[1]['crop']),
                     'amino': (singles[0]['amino'], singles[1]['amino']),
                     'coord': (singles[0]['coord'], singles[1]['coord']),
                     'pmtrx': (singles[0]['pmtrx'], singles[1]['pmtrx'])}]
        else:
            raise Exception('Not enough models.')
    else:
        return id, dimers


#Read pkls and dump pkls

def gen3_dump_pkl(outdir: str, file) -> str:
    id, results = gen3_read_pkl(file)
    path_out = os.path.join(outdir, f"gen3_{id}.pkl")
    with open(path_out, 'wb') as f:
        pkl.dump(results, f)
    return path_out, results

#preproc all pdb's...
#import pandas as pd

#print(fg_read_gz('1M1E.pdb.gz'))

def try_read_and_dump(path: str, i: int, n: int, outdir: str):
    try:
        print(f"\t[{i}/{n}] \t({path})\n", end='')
        out, prot = gen3_dump_pkl(outdir, path)
        print(f"\t[{i}/{n}] \t[{out}] \t{len(prot)} \t{'HOM' if prot[0]['homo'] else 'HET'} \t{'(s)' if prot[0]['singles'] else ''}\n",
              end='')
        return out
    # except Exception as e:
    #     print(f"\t[{i}/{n}] \t<{e}>\n", end='')
    except:
        print(f"\t[{i}/{n}] \t<ERROR>\n", end='')
    return None

if __name__ == '__main__':
    from task_utils import parallel_tasks_run_def
    import random as rnd
    import faulthandler
    import time
    
    faulthandler.enable(all_threads=False)
    print(f"faulthandler enabled? {faulthandler.is_enabled()}\nWait 3 s...")
    time.sleep(3)
    
    pkls = './temp/'
    temp = './temp_gen3/'
    tr_out = './train_dim.pkl'
    vl_out = './valid_dim.pkl'

    paths = [path for path in os.listdir(pkls) if path.split('.')[-1] == 'pkl']
    n = len(paths)
    print(f"Found {n} *.pkl")

    tasks = [[pkls + path, i, n, temp] for i, path in enumerate(paths)]
    prots = parallel_tasks_run_def(try_read_and_dump, tasks, task_name='Reading *.pkl...', num_workers=32, use_process=False)
    # prots = [try_read_and_dump(*tsk) for tsk in tasks]
    print('Tasks proceeded.')
    # print('Tasks skip')
    # prots = [temp + path for path in os.listdir(temp)]
    prots = [prot for prot in prots if prot is not None]
    prots.sort()
    
    total = []
    for i, path in enumerate(prots):
        if not path is None:
            with open(path, 'rb') as file:
                prot = pkl.load(file)
                total.extend(prot)
                print(f"\t[{i}] \tLoaded {path}")
    #total.sort()
    n = len(total)
    print(f"Total of {n} complexes.")

    rnd.seed(81)
    rnd.shuffle(total)
    tr_rate = 0.8
    tr_count = int(tr_rate * n)
    train, valid = total[0:tr_count], total[tr_count:]
    print(f"Train/Valid splitted {len(train)}/{len(valid)}")

    with open(tr_out, 'wb') as f:
        pkl.dump(train, f)
    with open(vl_out, 'wb') as f:
        pkl.dump(valid, f)
    print(f"Train dumped at: '{tr_out}'\nValid dumped at: '{vl_out}'")
