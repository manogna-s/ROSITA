import math
import operator
import numpy as np
import torch 
import torch.nn as nn
import os
import json
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.clip_tta_utils import compute_os_variance, accuracy, cal_auc_fpr, HM, get_ln_params
from torch.nn import functional as F
from utils.registry import METHODS_REGISTRY
from info_nce import InfoNCE

TPT_THRESHOLD = 0.1
ALIGN_THRESHOLD = 0.1




def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights.T


class TextResidue(nn.Module):
    def __init__(self, clip_weights):
        super(TextResidue, self).__init__()
        self.feat_dim, self.cate_num = clip_weights.shape
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, self.cate_num]).half().cuda(), requires_grad=True)
        
    def forward(self, x):
        new_clip_weights = x.clone() + self.residual
        new_clip_weights = F.normalize(new_clip_weights, dim=0)
        return new_clip_weights
    
    def reset(self):
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, self.cate_num]).half().cuda(), requires_grad=True)
        
        
class PositiveCacheResidue(nn.Module):
    def __init__(self, pos_cache_keys):
        super(PositiveCacheResidue, self).__init__()
        self.feat_dim, self.cache_size = pos_cache_keys.shape
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, self.cache_size]).half().cuda(), requires_grad=True)
        
    def forward(self, x):
        new_pos_cache_keys = x.clone() + self.residual
        new_pos_cache_keys = F.normalize(new_pos_cache_keys, dim=0)
        return new_pos_cache_keys


class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()

def InfoNCELoss(A, B):
    loss = InfoNCE(temperature=0.01, reduction='mean')
    return loss(A, B)


def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]
        return
    
    
def cache_key_value(image_features, cache, alpha, beta, clip_weights):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        all_classes = []
        for class_index in sorted(cache.keys()):
            num_items = len(cache[class_index])
            # Compute the prototype of the class
            image_prototype = torch.zeros_like(image_features)
            for item in cache[class_index]:
                image_prototype += item[0] / num_items
            cache_keys.append(image_prototype)
            cache_values.append(class_index)
            all_classes.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()
            
        return cache_keys, cache_values, all_classes
    
    
def compute_cache_logits(image_features, cache_keys, cache_values, alpha, beta, clip_weights):
    affinity = image_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits


@METHODS_REGISTRY.register()
def DPE(args, model, data_loader, classifiers=None):

    tta_method = args.tta_method
    tta_classifier = classifiers[args.classifier_type]
    
    log_dir_path = os.path.join(args.out_dir, args.model, args.desired, args.undesired, tta_method)
    os.makedirs(log_dir_path, exist_ok=True)
    log_file = open(f'{log_dir_path}/{tta_method}.txt', 'w')

    n_samples= {}
    n_samples['ALL'] = 0 
    n_samples['D'] = 0 
    n_samples['U_det'] = 0
    n_samples['D_total'] = 0
    n_samples['U_total'] = 0


    metrics_exp = {'Method':tta_method , 'AUC':0, 'FPR95':0, 'ACC_D':0, 'ACC_U':0, 'ACC_HM':0}
    ood_data = {'D': [], 'U': [], 'gt_idx': [], 'scores': []}

    top1, top5, n = 0, 0, 0
    scores_q = []
    scores_length = args.N_scores

    model.set_prompt_inits() 
    for nm, param in model.named_parameters():
        if "prompt_learner" not in nm:
            param.requires_grad_(False)
    

    model.eval()

    pos_cache = {}
    pos_cfg = {'enabled': True, 'shot_capacity': 3, 'alpha': 2, 'beta': 5.0}
    lr_cfg = {'text': 0.0006, 'image': 0.0006, 'align': 0.0}

    pos_enabled= pos_cfg['enabled']
    if pos_enabled:
        pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        
    clip_weights = tta_classifier.T
    clip_weights_global = tta_classifier.T.clone()
    num_avg = 0

    for i, (images, gt) in tqdm(enumerate(data_loader)):
        image = images[0]
        image, gt = image.cuda(), gt.cuda()
        ood_data['D'].append((gt<1000).item())
        ood_data['U'].append((gt>=1000).item())
        ood_data['gt_idx'].append(gt.item())
        
        
        #DPE
        clip_weights_local = clip_weights_global.clone().detach()
        text_residue = TextResidue(clip_weights_local)
        new_clip_weights = text_residue(clip_weights_local)
        
        with torch.cuda.amp.autocast():
            image_features_x = model.encode_image(image)
            image_features_x = image_features_x / image_features_x.norm(dim=-1, keepdim=True)
            clip_logits = (image_features_x @ new_clip_weights) * 100
            
            entropy = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])
            prop_entropy = get_entropy(entropy, new_clip_weights)
            
            if pos_enabled:
                entropy = get_entropy(entropy, new_clip_weights)
                update_cache(pos_cache, pred, [image_features_x, entropy], pos_params['shot_capacity'])
                pos_cache_keys, pos_cache_values, all_classes = cache_key_value(image_features_x, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
                pos_cache_residue = PositiveCacheResidue(pos_cache_keys)
            
            new_clip_weights = text_residue(clip_weights_local)
            final_logits = clip_logits.clone()
            if pos_enabled and pos_cache:
                new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                final_logits += compute_cache_logits(image_features_x, new_pos_cache_keys, pos_cache_values, pos_params['alpha'], pos_params['beta'], clip_weights)
                loss = avg_entropy(final_logits)
                # alignment loss
                image2text_loss = InfoNCELoss(new_pos_cache_keys.T, new_clip_weights[:, all_classes].T)
                loss += image2text_loss * lr_cfg['align']
            else:
                loss = avg_entropy(final_logits)
            
            lr_text = lr_cfg['text']
            lr_image = lr_cfg['image']
            if pos_enabled and pos_cache:
                optimizer = torch.optim.AdamW([
                    {'params': text_residue.parameters(), 'lr': lr_text, 'eps': 1e-3, 'weight_decay': 1e-1},
                    {'params': pos_cache_residue.parameters(), 'lr': lr_image, 'eps': 1e-3, 'weight_decay': 1e-1}
                    ])
            else:
                optimizer = torch.optim.AdamW([
                    {'params': text_residue.parameters(), 'lr': lr_text, 'eps': 1e-3, 'weight_decay': 1e-1}
                    ])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Actual inference
            text_residue.eval()
            if pos_enabled and pos_cache:
                pos_cache_residue.eval()
            with torch.no_grad():
                new_clip_weights = text_residue(clip_weights_local)
                image_features = model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                clip_logits = (image_features @ new_clip_weights) * 100
                final_logits = clip_logits.clone()
                if pos_enabled and pos_cache:
                    new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                    final_logits += compute_cache_logits(image_features, new_pos_cache_keys, pos_cache_values, pos_params['alpha'], pos_params['beta'], clip_weights) 
            loss = avg_entropy(final_logits)
            if get_entropy(loss, tta_classifier) < 0.1:
                num_avg += 1
                clip_weights_global = clip_weights_global * (num_avg / (num_avg + 1)) + new_clip_weights * (1 / (num_avg + 1))
            
            logits = final_logits
            maxlogit_tta, pred_tta = logits.max(1)

        threshold_range = np.arange(0,100,1)
        ood_data['scores'].extend(maxlogit_tta.tolist())
        scores_q.extend(maxlogit_tta.tolist())
        scores_q = scores_q[-scores_length:]
        criterias = [compute_os_variance(np.array(scores_q), th) for th in threshold_range]
        best_thresh = threshold_range[np.argmin(criterias)]

        W_curr, S_curr = gt<1000, gt>=1000
        W_pred, S_pred = maxlogit_tta >= best_thresh, maxlogit_tta < best_thresh

        # metrics
        n_samples['D_total'] += torch.sum(W_curr).item()
        n_samples['U_det'] += torch.sum(S_pred[S_curr]).item()
        n_samples['U_total'] += torch.sum(S_curr).item()

        if W_pred[0].item():
            n_samples['D'] += torch.sum(gt[gt<1000]==pred_tta[gt<1000]).item()
        n_samples['ALL'] += torch.sum(gt[gt<1000]==pred_tta[gt<1000]).item()

        if (i+1) %1000 == 0:
            acc_w = n_samples['D']/n_samples['D_total']
            acc_s = n_samples['U_det']/n_samples['U_total']
            acc_hm = HM(acc_w, acc_s)
            status_log = f'\nStep {i}:   ACC_D: {acc_w}; ACC_U: {acc_s}; ACC_HM: {acc_hm}'
            print(status_log)
            log_file.write(status_log)

    metrics_exp['ACC_D'] = n_samples['D']/n_samples['D_total']
    metrics_exp['ACC_U'] = n_samples['U_det']/n_samples['U_total']

    ood_data['scores'] = np.array(ood_data['scores'])
    metrics_exp['AUC'], metrics_exp['FPR95'] = cal_auc_fpr(ood_data['scores'][ood_data['D']], ood_data['scores'][ood_data['U']])
    metrics_exp['ACC_HM'] = HM(metrics_exp['ACC_D'], metrics_exp['ACC_U'])

    print(f'\n\n')
    print(args.desired, args.undesired, tta_method)
    print(f'Metrics: {metrics_exp}\n')

    df_metrics = pd.DataFrame([metrics_exp])
    df_metrics.to_csv(f'{log_dir_path}/{args.tta_method}_results.csv', index=False)  

    plt.hist(ood_data['scores'][ood_data['D']], bins=threshold_range, label='Desired', alpha=0.5)
    plt.hist(ood_data['scores'][ood_data['U']], bins=threshold_range, label='Undesired', alpha=0.5)
    plt.legend()
    plt.savefig(f'{log_dir_path}/{args.tta_method}_scores_hist.jpg')

    return metrics_exp


