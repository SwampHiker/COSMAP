#**********************************************#
# Evaluating first stage network.              #
#                                              #
#**********************************************#

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import cosma_algebra as ca
import torch

train = './train.pkl'
valid = './valid.pkl'
pipkl = './gen1_log/gen1_pipe.pkl'
outdr = './gen1_log/prediction.pkl'

def get_stat(pipeline, data, show=True):
    print(f"Data size: {len(data)}")
    results = []
    for i, prot in enumerate(data):
        pred = {'id': prot['id'], 'idx': prot['idx'], 'crop': prot['crop'],  #+ Lengths in L-matrix
                'cuts': np.sum(ca.length(ca.diff(prot['coord'])) > 4.5),
                'rmsd': 0, 'ssim0': 0, 'ssim1': 0, 'ssim_d': 0, 'error': 0}
        try:
            with torch.no_grad():
                amino = ca.amino_list_to_array_stack(prot['amino'], False).astype(np.float32)
                amino = torch.tensor(np.array([amino]), device='cuda')
                cosma = ca.cosma_from_pmtrx(prot['pmtrx'], prot['pmtrx'])
                cosma = torch.tensor(np.array([[cosma]]), device='cuda')

                cosma_pred = pipeline.model(amino.mT)
                pred['ssim0'] = float(pipeline.loss(cosma_pred, cosma).detach().cpu())
                lmtrx = ca.get_lmtrx_from_cosma_E(cosma_pred.detach().cpu().numpy()[0, 0])
                pmtrx = ca.normalize(lmtrx)
                cosma_ref = ca.cosma_from_pmtrx(pmtrx, pmtrx)
                cosma_ref = torch.tensor(np.array([[cosma_ref]]), device='cuda')
                pred['ssim1'] = float(pipeline.loss(cosma_ref, cosma).detach().cpu())
                pred['ssim_d'] = float(pipeline.loss(cosma_ref, cosma_pred).detach().cpu())

                lmtrx = pmtrx * 3.8
                coord_pred = ca.align_lmtrx_to_coord(lmtrx, prot['coord'])
                pred['rmsd'] = float(ca.get_rmsd(coord_pred, prot['coord']))
            print(f"{i}: {prot['id']} \t({prot['crop']})\t/{pred['cuts']}/ \tRMSD = {pred['rmsd']} A*")
        except:
            pred['error'] = 1
            print(f"{i}: {prot['id']} \t({prot['crop']})\t/{pred['cuts']}/ \tERROR")
        results.append(pred)

    print('Calculation end.')
    rmsds = [pred['rmsd'] for pred in results if pred['error'] == 0]
    print(f"Non-error predictions: {len(rmsds)}\nMean RMSD: {np.mean(rmsds)}")
    if show:
        plt.hist([r for r in rmsds if r < 45], 100, histtype='step')
        plt.show()
    return results


def dump_csv(data: list[dict], out):
    names = list(data[0].keys())
    out.write(','.join(names) + '\n')
    for prot in data:
        out.write(','.join([str(prot[name]) for name in names]) + '\n')


if __name__ == '__main__':
    with open(pipkl, 'rb') as file:
        pipe = pkl.load(file).cuda().eval() #eval
        print('Pipeline loaded.')
    
    with open(train, 'rb') as file:
        print('TRAIN')
        train_result = get_stat(pipe, pkl.load(file))

    with open(valid, 'rb') as file:
        print('VALID')
        valid_result = get_stat(pipe, pkl.load(file))

    with open(outdr, 'wb') as file:
        pkl.dump((train_result, valid_result), file)
        print('Results dumped.')
