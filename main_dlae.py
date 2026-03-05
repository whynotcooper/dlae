import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn.functional as F
import operator
import torch.nn as nn
from info_nce import InfoNCE
#from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import clip
from utils import *
import os
import open_clip

#os.environ["WANDB_MODE"] = "offline"   # Offline mode (do not modify this line)
import numpy as np
import matplotlib.pyplot as plt


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs',
                        help='settings of DPE on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', default=True, action='store_true',
                        help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, default='dtd',
                        help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='../dataset',
                        help='Path to the datasets directory. Default is ../data/')
    parser.add_argument('--backbone', dest='backbone', default='ViT-B/16', type=str,
                        choices=['RN50', 'ViT-B/16', 'SigLIP', 'OpenCLIP'],
                        help='CLIP model backbone to use: RN50 or ViT-B/16.')
    parser.add_argument('--coop', dest='coop', default=False, action='store_true',
                        help='Whether you want to use CoOp weights for initialization.')
    print("Parsing arguments...")
    print(parser)
    args = parser.parse_args()

    return args


def InfoNCELoss(A, B):
    loss = InfoNCE(temperature=0.01, reduction='mean')
    return loss(A, B)


def visualize_cache(cache, iter):
    # t-SNE visualization of cache features
    with torch.no_grad():
        cache_features = []
        cache_labels = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_features.append(item[0].reshape(-1))
                cache_labels.append(class_index)
        cache_features = torch.stack(cache_features, dim=0)
        cache_labels = torch.Tensor(cache_labels).to(torch.int64)
        cache_features = F.normalize(cache_features, dim=1)
        cache_features = cache_features.cpu().numpy()
        cache_labels = cache_labels.cpu().numpy()
        tsne = TSNE(n_components=2)
        print(cache_features.shape)
        cache_features_fit = tsne.fit_transform(cache_features)

        # Assign different colors to different cache_labels
        colors = [
            '#00429d',  # Strong Blue
            '#93003a',  # Deep Red
            '#007d34',  # Vivid Green
            '#ff6800',  # Vivid Orange
            '#e30022',  # Bright Red
            '#a6bdd7',  # Light Periwinkle
            '#ffcc00',  # Vivid Yellow
            '#540d6e',  # Dark Violet
            '#7f180d',  # Dark Red
            '#00939c',  # Cyan Process
            '#5f3c99',  # Purplish Blue
            '#ff4a46',  # Bright Red-Orange
            '#8f0075',  # Strong Purple
            '#ff3c38',  # Bright Red
            '#83a697',  # Muted Cyan
            '#1e96be',  # Strong Cyan
            '#d9e021',  # Vivid Lime Green
            '#f18d05',  # Rich Orange
            '#f6e120',  # Bright Yellow
            '#8f2d56',  # Strong Rose
            '#006837',  # Dark Green
            '#e7298a',  # Bright Pink
            '#ce1256',  # Dark Pink
            '#01665e',  # Dark Teal
            '#dfc27d',  # Pale Gold
            '#35978f',  # Muted Teal
            '#bf812d',  # Mustard Brown
            '#543005',  # Dark Brown
            '#8c510a',  # Light Brown
            '#80cdc1',  # Soft Turquoise
        ]
        colors_others = 'gray'
        figure, ax = plt.subplots(1, 1, dpi=600, figsize=(5, 5))
        patch = ax.patch
        patch.set_color("#f5f5f5")
        ax.tick_params(axis='both',  # Changes apply to both x and y axes
                       which='both',  # Apply changes to both major and minor ticks
                       bottom=False,  # No ticks along the bottom edge
                       top=False,  # No ticks along the top edge
                       left=False,  # No ticks along the left edge
                       right=False,  # No ticks along the right edge
                       labelbottom=False,  # No labels along the bottom edge
                       labelleft=False)  # No labels along the left edge
        plt.grid(color='w', zorder=0, linewidth=2)
        plt.gca().spines['bottom'].set_color('gray')
        plt.gca().spines['left'].set_color('gray')
        plt.gca().spines['top'].set_color('gray')
        plt.gca().spines['right'].set_color('gray')
        # In Food-101, we have 101 classes
        for i in range(101):
            if i < 30:
                plt.scatter(cache_features_fit[cache_labels == i, 0], cache_features_fit[cache_labels == i, 1],
                            c=colors[i], s=15, marker='x', zorder=5)
            else:
                plt.scatter(cache_features_fit[cache_labels == i, 0], cache_features_fit[cache_labels == i, 1],
                            c=colors_others, s=5, zorder=5)
        save_path = 'fig/cache_features_iter_{}.png'.format(iter)
        plt.savefig(save_path)
        plt.close()


def cache_key_value(image_features, cache, alpha, beta, clip_weights):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        all_classes = []
        for class_index in sorted(cache.keys()):
            num_items = len(cache[class_index])
            # Compute the class prototype
            image_prototype = torch.zeros_like(image_features)
            count = 0
            for item in cache[class_index]:
                if count <= 2:
                    image_prototype += item[0] / num_items
                    count = count + 1
            cache_keys.append(image_prototype)
            cache_values.append(class_index)
            all_classes.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        cache_values = (
            F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()

        return cache_keys, cache_values, all_classes


def compute_cache_logits(image_features, cache_keys, cache_values, alpha, beta, clip_weights):
    affinity = image_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits


def exp_loss(x, time, alpha):
    # Ensure x, time, alpha are all PyTorch tensors
    x = torch.tensor(x) if not isinstance(x, torch.Tensor) else x
    time = torch.tensor(time) if not isinstance(time, torch.Tensor) else time
    alpha = torch.tensor(alpha) if not isinstance(alpha, torch.Tensor) else alpha

    return x * torch.exp(time * alpha)

def compute_cache_logits(image_features, cache_keys, cache_values, alpha, beta, clip_weights):
    affinity = image_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits


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

def Init_conf(cache_size, my_device):
    conf_count = torch.zeros((cache_size,), device=my_device, requires_grad=False)
    return conf_count

def Init_confcount(cache_size, my_device):
    conf_count = torch.zeros((cache_size,), device=my_device, requires_grad=False)
    return conf_count

def update_mean_conf(conf_count, pred, count, conf):
    # Used to update the confidence
    conf_count[pred] = (conf_count[pred] * count[pred] + conf) / (count[pred] + 1)
    count[pred] = count[pred] + 1

def check_doubt(now_weights, weights0, image_featurs):
    if image_featurs @ now_weights < image_featurs @ weights0:
        return True
    else:
        return False

def update_cache(cache, pred, features_loss_time_doubt, time_now, shot_capacity,
                 include_prob_map=False, alpha=200, clip_weights=None, weights0=None):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        # If include_prob_map is False, item is features_loss_time_doubt;
        # otherwise item is the first two elements plus the third element (prob map)
        item = features_loss_time_doubt if not include_prob_map else features_loss_time_doubt[:2] + [features_loss_time_doubt[2]]
        # If pred exists in cache
        if pred in cache:
            # If the number of items under pred is less than shot_capacity
            if len(cache[pred]) < shot_capacity:
                # Append item to the cache for this pred
                cache[pred].append(item)
            else:
                # Traverse items under this pred
                for i in range(len(cache[pred])):
                    # If the last element in this cache item is True
                    # Check doubt

                    if cache[pred][i][-1] == True:
                        doubt = check_doubt(clip_weights.mT[pred], weights0.mT[pred], cache[pred][i][0])

                        # If doubt is True
                        if doubt:
                            # Compute exp_loss
                            cache[pred][i][1] = exp_loss(cache[pred][i][1], time_now - cache[pred][i][2], alpha)
                            # Update time
                            cache[pred][i][2] = time_now

                # Sort cache[pred] by the second element
                cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
                # If the new item's loss is smaller than the largest loss in cache[pred]
                if features_loss_time_doubt[1] < cache[pred][-1][1]:
                    # Replace the last element with the new item
                    cache[pred][-1] = item
                    # Sort again by the second element
                    cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))

        else:
            # If pred is not in cache, create a new list with this item
            cache[pred] = [item]
        return


def run_test_dlae(pos_cfg, lr_cfg, loader, clip_model, clip_weights, dataset_name, alpha, alpha1, epoch):
    with torch.cuda.amp.autocast():
        pos_cache, accuracies = {}, []

        my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check if GPU is available
        # Unpack all hyperparameters
        pos_enabled = pos_cfg['enabled']

        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}

        clip_weights_global = clip_weights.clone()
        num_avg = 0

        cache_size = clip_weights.shape[1]
        # Used to count the number of pseudo-labels for each class
        counter = Init_counter(cache_size, my_device)
        # Used to count the total number
        count = torch.tensor(cache_size, device=my_device)
        # Used to record the average confidence for each class
        conf_counter = Init_conf(cache_size, my_device)
        # Used to count how many times each class is predicted (one less than counter)
        conf_count = Init_confcount(cache_size, my_device)
        weights0 = clip_weights.clone()

        # Test-time adaptation
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            clip_weights_local = clip_weights_global.clone().detach()
            text_residue = TextResidue(clip_weights_local)
            new_clip_weights = text_residue(clip_weights_local)
            # We need to adjust
            image_features_x, clip_logits, entropy, prob_map, pred, doubt, features_all, entropy1 = get_clip_logits(
                images, clip_model, new_clip_weights, counter, count, alpha, conf_counter, alpha1
            )
            target = target.cuda()
            counter = update_counter(counter, pred)
            count = count + 1
            update_mean_conf(conf_counter, pred, conf_count, prob_map.squeeze(0)[pred].detach().cpu().item())

            if pos_enabled:
                update_cache(
                    pos_cache, pred, [image_features_x, entropy1, i, True],
                    i, pos_params['shot_capacity'],
                    alpha=epoch, clip_weights=new_clip_weights, weights0=weights0
                )

                # visualize_cache(pos_cache, i)

                pos_cache_keys, pos_cache_values, all_classes = cache_key_value(
                    image_features_x, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights
                )

                # visualize_cache(pos_cache, i)

                pos_cache_residue = PositiveCacheResidue(pos_cache_keys)

            steps = 1  # Update step, set to 1 by default
            for j in range(steps):
                new_clip_weights = text_residue(clip_weights_local)
                final_logits = clip_logits.clone()
                if pos_enabled and pos_cache:
                    new_pos_cache_keys = pos_cache_residue(pos_cache_keys)

                    final_logits += compute_cache_logits(
                        image_features_x, new_pos_cache_keys, pos_cache_values,
                        pos_params['alpha'], pos_params['beta'], clip_weights
                    )
                    loss = avg_entropy(final_logits)
                    # Alignment loss
                    image2text_loss = InfoNCELoss(new_pos_cache_keys.T, new_clip_weights[:, all_classes].T)
                    loss += image2text_loss * lr_cfg['align']
                else:
                    loss = avg_entropy(final_logits)

                lr_text = lr_cfg['text']
                lr_image = lr_cfg['image']
                if pos_enabled and pos_cache:
                    optimizer = torch.optim.AdamW([
                        {'params': text_residue.parameters(), 'lr': lr_text, 'eps': 1e-3, 'weight_decay': 1e-1},
                        {'params': pos_cache_residue.parameters(), 'lr': lr_text, 'eps': 1e-3, 'weight_decay': 1e-1},
                    ])
                else:
                    optimizer = torch.optim.AdamW([
                        {'params': text_residue.parameters(), 'lr': lr_text, 'eps': 1e-3, 'weight_decay': 1e-1}
                    ])

                optimizer.zero_grad()
                if j == steps - 1:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
                optimizer.step()

            # Actual inference
            text_residue.eval()
            if pos_enabled and pos_cache:
                pos_cache_residue.eval()
            with torch.no_grad():
                new_clip_weights = text_residue(clip_weights_local)
                if dataset_name == 'A':
                    image_features, clip_logits, _, _, _, _ = get_clip_logits_2(
                        features_all, clip_model, new_clip_weights, counter, count, alpha, conf_counter, alpha1
                    )
                else:
                    image_features, clip_logits, _, _, _, _ = get_clip_logits_2(
                        features_all[0], clip_model, new_clip_weights, counter, count, alpha, conf_counter, alpha1
                    )
                final_logits = clip_logits.clone()
                if pos_enabled and pos_cache:
                    new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                    final_logits += compute_cache_logits(
                        image_features, new_pos_cache_keys, pos_cache_values,
                        pos_params['alpha'], pos_params['beta'], clip_weights
                    )

                acc = cls_acc(final_logits, target.cuda())

                accuracies.append(acc)
                wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)}, commit=True)

                loss = avg_entropy(final_logits)

                if get_entropy(loss, clip_weights) < 0.1:
                    num_avg += 1
                    clip_weights_global = clip_weights_global * (num_avg / (num_avg + 1)) + new_clip_weights * (
                        1 / (num_avg + 1))

            if i % 1000 == 0:
                print("---- DPE's test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))
    print("---- DPE's test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))

    return sum(accuracies) / len(accuracies)


def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    if args.backbone == 'RN50' or args.backbone == 'ViT-B/16':
        clip_model, preprocess = clip.load(args.backbone)
    elif args.backbone == 'SigLIP':
        clip_model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP')
    elif args.backbone == 'OpenCLIP':
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            'hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
        clip_model = clip_model.to('cuda')

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"

    # Run DPE on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        # Set random seed
        random.seed(1)
        torch.manual_seed(1)
        print(f"Processing {dataset_name} dataset.")

        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        print(args.coop)
        print(args.backbone)
        if args.wandb:
            run_name = f"{dataset_name}"
            run = wandb.init(project="DLAE", config=cfg, group=group_name, name=run_name)
        test_loader, classnames, template, cupl_path = build_test_data_loader(dataset_name, args.data_root, preprocess)

        if args.backbone == 'RN50':
            clip_weights = clip_classifier(classnames, template, cupl_path, clip_model, args.coop, args.backbone)
            acc = run_test_dlae(
                cfg['positive'], cfg['learning_rate'], test_loader, clip_model, clip_weights,
                dataset_name, cfg['alpha'][0], cfg['alpha1'][0], cfg['epoch'][0]
            )

        if args.backbone == 'ViT-B/16':
            clip_weights = clip_classifier(classnames, template, cupl_path, clip_model, args.coop, dataset_name, args.backbone)
            acc = run_test_dlae(
                cfg['positive'], cfg['learning_rate'], test_loader, clip_model, clip_weights,
                dataset_name, cfg['belta'][0], cfg['alpha1'][1], cfg['epoch'][1]
            )

        if args.wandb:
            wandb.log({f"{dataset_name}": acc})
            run.finish()


# Ensure the entry point
if __name__ == "__main__":
    main()
