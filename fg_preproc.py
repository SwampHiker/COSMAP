#**********************************************#
# The preproccessing of tertiery structures.   #
#   (creating pkl out of pdbs)                 #
#**********************************************#

import os
import gzip as gz
import numpy as np
# import Bio as bio
import pickle as pkl
import Bio.PDB as pdb
import cosma_algebra as ca


def fg_read_pdb(id: str, file) -> list[dict]:
    """Reads all monomers and pkls it's aminos and cosmas."""
    parser = pdb.PDBParser(QUIET = True)
    struct = parser.get_structure(id, file)
    models = []
    for idx, model in enumerate(list(struct.get_models())):
        chains = list(model.get_chains())
        for chain in chains:
            resid = []
            coord = []
            for residue in chain.get_residues():
                resname = residue.resname
                for atom in residue.get_atoms():
                    if atom.name == 'CA':
                        resid.append(resname)
                        coord.append(atom.coord)
                        break
            if not resid or not coord:
                continue
            coord = np.transpose(np.array(coord))
            crop = len(resid)
            pmtrx = ca.get_pmtrx(coord)
            if np.any(np.isnan(pmtrx)): #fight the NaNs
                continue
            models.append({'id': id, 'idx': idx, 'crop': crop, 'amino': resid, 'coord': coord, 'pmtrx': pmtrx})
    if len(models) == 0:
        raise Exception('No models')
    return models


#Обработка и дамп файла
def fg_dump_pdb(outdir: str, id: str, file) -> str:
    results = fg_read_pdb(id, file)
    path_out = os.path.join(outdir, f"dump_{id}.pkl")
    with open(path_out, 'wb') as f:
        pkl.dump(results, f)
    return path_out


#Прочитка gz и дамп файла
def fg_dump_gz(outdir: str, gz_path: str) -> str:
    id = os.path.basename(gz_path).split('.')[0]
    with gz.open(gz_path, 'rt') as file:
        return fg_dump_pdb(outdir, id, file)

def fg_read_gz(gz_path: str) -> list[dict]:
    id = os.path.basename(gz_path).split('.')[0]
    with gz.open(gz_path, 'rt') as file:
        return fg_read_pdb(id, file)
    

#preproc all pdb's...
#import pandas as pd

#print(fg_read_gz('1M1E.pdb.gz'))

def try_read_and_dump(path: str, i: int, n: int, outdir: str):
    try:
        print(f"\t[{i}/{n}] \t({path})\n", end='')
        prot = fg_dump_gz(outdir, path)
        print(f"\t[{i}/{n}] \t[{prot}]\n", end='')
        return prot
    except:
        print(f"\t[{i}/{n}] \t<ERROR>\n", end='')
        return None

if __name__ == '__main__':
    from task_utils import parallel_tasks_run_def
    import random as rnd
    
    gzs = './NEW PROTS/'
    temp = './temp/'
    tr_out = './train.pkl'
    vl_out = './valid.pkl'

    paths = [path for path in os.listdir(gzs) if path.split('.')[-1] == 'gz']
    n = len(paths)
    print(f"Found {n} *.gz")

    tasks = [[gzs + path, i, n, temp] for i, path in enumerate(paths)]
    prots = parallel_tasks_run_def(try_read_and_dump, tasks, task_name='Reading *.pdb.gz...', num_workers=32, use_process=False)
    print('Tasks proceeded.')
    # print('Tasks skip')
    prots = [temp + path for path in os.listdir(temp)]
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
    print(f"Total of {n} chains")

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
