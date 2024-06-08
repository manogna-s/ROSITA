import numpy as np
import torch 
import os
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.clip_tta_utils import compute_os_variance, accuracy, cal_auc_fpr, HM
from utils.registry import METHODS_REGISTRY

@METHODS_REGISTRY.register()
def ZSEval(args, model, data_loader, classifiers):

    classifier = classifiers[args.classifier_type]
    tta_method = f'{args.tta_method}'   
    log_dir_path = os.path.join(args.out_dir, args.model, args.weak_OOD, args.strong_OOD, tta_method)
    os.makedirs(log_dir_path, exist_ok=True)


    n_samples= {}
    n_samples['ALL'] = 0
    n_samples['W'] = 0
    n_samples['S_det'] = 0
    n_samples['W_total'] = 0
    n_samples['S_total'] = 0


    metrics_exp = {'Method':tta_method , 'AUC':0, 'FPR95':0, 'ACC_W':0, 'ACC_S':0, 'ACC_HM':0}
    ood_data = {'W': [], 'S': [], 'gt_idx': [], 'ood_scores': []}


    top1, top5, n = 0, 0, 0
    scores_q = []
    scores_length = args.N_scores
    
    
    for i, (image, gt) in tqdm(enumerate(data_loader)):
        if isinstance(image,list):
            image = image[0].cuda()
        else:
            image = image.cuda()
        image, gt = image.cuda(), gt.cuda()
        ood_data['W'].append((gt<1000).item())
        ood_data['S'].append((gt>=1000).item())
        ood_data['gt_idx'].append(gt.item())

        # Base eval

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                imf_norm = model.encode_image(image)
                imf_norm = imf_norm/imf_norm.norm(dim=-1, keepdim=True)
                logits_txt =  (imf_norm @ classifier.T)
                scores_txt = (logits_txt * 100).softmax(1)
                maxlogit_txt, _ = logits_txt.max(1)

                conf_txt, pred_txt = torch.max(scores_txt, dim=1)

                threshold_range = np.arange(0,1,0.01)
                ood_data['ood_scores'].extend(maxlogit_txt.tolist())
                scores_q.extend(maxlogit_txt.tolist())
                scores_q = scores_q[-scores_length:]
                criterias = [compute_os_variance(np.array(scores_q), th) for th in threshold_range]
                best_thresh = threshold_range[np.argmin(criterias)]

        W_curr, S_curr = gt<1000, gt>=1000
        W_pred, S_pred = maxlogit_txt >= best_thresh, maxlogit_txt < best_thresh

        # metrics
        n_samples['W_total'] += torch.sum(W_curr).item()
        n_samples['S_det'] += torch.sum(S_pred[S_curr]).item()
        n_samples['S_total'] += torch.sum(S_curr).item()

        if W_pred[0].item():
            n_samples['W'] += torch.sum(gt[gt<1000]==pred_txt[gt<1000]).item()
        n_samples['ALL'] += torch.sum(gt[gt<1000]==pred_txt[gt<1000]).item()

        if (i+1) %1000 == 0:
            acc_w = n_samples['W']/n_samples['W_total']
            acc_s = n_samples['S_det']/n_samples['S_total']
            acc_hm = HM(acc_w, acc_s)
            print(f'\nStep {i}:   ACC_W: {acc_w}; ACC_S: {acc_s}; ACC_HM: {acc_hm}')


    metrics_exp['ACC_W'] = n_samples['W']/n_samples['W_total']
    metrics_exp['ACC_S'] = n_samples['S_det']/n_samples['S_total']

    ood_data['ood_scores'] = np.array(ood_data['ood_scores'])
    metrics_exp['AUC'], metrics_exp['FPR95'] = cal_auc_fpr(ood_data['ood_scores'][ood_data['W']], ood_data['ood_scores'][ood_data['S']])
    metrics_exp['ACC_HM'] = HM(metrics_exp['ACC_W'], metrics_exp['ACC_S'])

    print(f'\n\n')
    print(args.weak_OOD, args.strong_OOD, tta_method)
    print(f'Final Metrics: {metrics_exp}\n')

    df_metrics = pd.DataFrame([metrics_exp])
    df_metrics.to_csv(f'{log_dir_path}/{args.tta_method}_results.csv', index=False)  

    plt.hist(ood_data['ood_scores'][ood_data['W']], bins=threshold_range, label='Weak OOD', alpha=0.5)
    plt.hist(ood_data['ood_scores'][ood_data['S']], bins=threshold_range, label='Strong OOD', alpha=0.5)
    plt.legend()
    plt.savefig(f'{log_dir_path}/{args.tta_method}_scores_hist.jpg')

    return metrics_exp