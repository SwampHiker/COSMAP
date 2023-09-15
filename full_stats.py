#**********************************************#
# Evaluating second stage network.             #
#                                              #
#**********************************************#

import torch
import numpy as np
import pickle as pkl
import cosma_algebra as ca
import full_pipeline as full
from task_utils import parallel_tasks_run_def


train_set = full.Full_Dataset('./train_dim.pkl')
valid_set = full.Full_Dataset('./valid_dim.pkl')

train_set.set_crop(64)
valid_set.set_crop(64)

with open('./full_log/full_pipe.pkl', 'rb') as f:
    full_pipe = pkl.load(f).cuda()


def proceed(i, dataset):
    pmtrx1, pmtrx2, amino1, amino2, cv1, cv2 = dataset.__getitem__(i, False,
                                                                   True)
    with torch.no_grad():
        # on GPU
        pmtrx1_tens = torch.tensor(np.array([pmtrx1]), device='cuda')
        pmtrx2_tens = torch.tensor(np.array([pmtrx2]), device='cuda')
        amino1_tens = torch.tensor(np.array([amino1]), device='cuda')
        amino2_tens = torch.tensor(np.array([amino2]), device='cuda')
        cv1_tens = torch.tensor(np.array([cv1]), device='cuda')
        cv2_tens = torch.tensor(np.array([cv2]), device='cuda')

        code_1 = full_pipe.model.encode_amino(amino1_tens)
        code_2 = full_pipe.model.encode_amino(amino2_tens)

        true_tens = full_pipe.prepare_true_tensor(pmtrx1_tens, pmtrx2_tens,
                                                  cv1_tens, cv2_tens)
        pred_tens = full_pipe(amino1_tens, code_1, amino2_tens, code_2)

        loss_raw = float(full_pipe.loss(pred_tens,
                                        true_tens).detach().cpu().numpy())

    # on CPU
    cosma_pred, cv1_pred, cv2_pred = pred_tens.detach().cpu().numpy()[0]

    prot = dataset.load_item(i)

    honesty = ca.rand_gen_s().astype(np.float32)
    pred_dict = ca.full_model(prot['coord'][0], honesty @ prot['coord'][1],
                              cosma_pred, cv1_pred, cv2_pred, prot['homo'])

    comparison = ca.compare_models(prot['coord'][0], prot['coord'][1],
                                   pred_dict['coord'][1], prot['homo'])

    return {'id': prot['id'], 'homo': prot['homo'], 'crop': prot['crop'],
            'ssim_raw': loss_raw, 'cosma_dev': pred_dict['cosma_dev'],
            'cv1_dev': pred_dict['cv_dev'][0],
            'cv2_dev': pred_dict['cv_dev'][1], **comparison}


if __name__ == '__main__':
    dataset_name = 'TRAIN'
    dataset = train_set

    def out(i):
        try:
            result = proceed(i, dataset)
            print(f"{dataset_name} [{i}/{len(dataset)}]" +
                  f" RMSD {result['rmsd']:.2f}   \t" +
                  f"   \tSSIM {result['ssim_raw']:.2f}   \t" +
                  ('1' if result['rmsd'] < 10 else ' ') + '\t' +
                  (' HOM' if result['homo'] else ' HET') + '\n', end='')
            return result
        except Exception:
            print(f"{dataset_name} [{i}/{len(dataset)}] ERROR\n", end='')
            return None

    # train_result = parallel_tasks_run_def(out, range(len(dataset)),
    #                                       num_workers=8, use_process=False)

    train_result = []

    dataset_name = 'VALID'
    dataset = valid_set

    valid_result = parallel_tasks_run_def(out, range(len(dataset)),
                                          num_workers=8, use_process=False)

    # for i in range(len(valid_set)):
    #     result = proceed(i, valid_set)
    #     valid_result.append(result)
    #     print(f"VALID [{i}/{len(valid_set)}] (RMSD_A {
    # result['rmsd_a']:.2f})" +
    #           f"   \t(RMSD_P {result['rmsd_p']:.2f})   \t" +
    #           ('1' if result['rmsd_a'] < 10 else '_') +
    #           ('2' if result['rmsd_p'] < 10 else '_') +
    #           (' HOM\n' if result['homo'] else ' HET\n'), end='')

    with open('predict_full_old_bump_valid.pkl', 'wb') as f:
        pkl.dump((train_result, valid_result), f)
