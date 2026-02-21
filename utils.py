def exp_decay(x, alpha=2.5, beta=0):
    return torch.exp(-alpha * x + beta)

def logits_adjustment(logits, dif, counter, count, alpha):
    prob = counter / count
    logits = logits * exp_decay(x=prob, alpha=-alpha * dif + alpha)
    return logits

def Init_counter(cache_size, device='cpu'):
    counter = torch.ones(cache_size, dtype=torch.int32, device=device)
    return counter

def update_counter(counter, pred):
    counter[pred] += 1
    return counter
def cos_sim(x, y):
    if x.dim()==2: 
        return F.cosine_similarity(x, y,dim=1)
    else:
        return F.cosine_similarity(x, y,dim=0)
def get_clip_logits_2(images, clip_model, clip_weights,counter,count,alpha=1,mean_loss=None,alpha1=None):
    # with torch.no_grad():
    if isinstance(images, list):
        images = torch.cat(images, dim=0).cuda()
    else:
        images = images.cuda()
    
    # Change 3D tensor to 4D tensor
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    with torch.no_grad():
      image_features = clip_model.encode_image(images)
      image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    doubt=False
    clip_logits = 100. * image_features @ clip_weights
    image_features_all=image_features.clone()
    if image_features.size(0) > 1:
        batch_entropy2 = softmax_entropy(clip_logits)
        selected_idx2 = torch.argsort(batch_entropy2, descending=False)[:int(batch_entropy2.size()[0] * 0.1)]
        output2 = clip_logits[selected_idx2]

        prob_map2 = output2.softmax(1).mean(0).unsqueeze(0)
        pred2 = int(output2.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        
        #caculate the pred2 and loss 2
        # logits[n]=logtis[n]*(-a*(1+conf_past-conf_now)*prob[n])  如果现在conf_now小，加大 logits adjustment力度 
        clip_logits=logits_adjustment(clip_logits,prob_map2.squeeze(0)[pred2].detach().cpu().item()-mean_loss[pred2].detach().cpu().item(),counter,count,alpha)
        batch_entropy = softmax_entropy(clip_logits)
        selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
        output = clip_logits[selected_idx]
        image_features = image_features[selected_idx].mean(0).unsqueeze(0)
        clip_logits = output.mean(0).unsqueeze(0)
        prob_map = output.softmax(1).mean(0).unsqueeze(0)
        loss = avg_entropy(output) 
        pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())  
    else:

        prob_map2 = clip_logits.softmax(1)
        pred2 = int(clip_logits.topk(1, 1, True, True)[1].t()[0])
        output2=clip_logits
        # caculate the pred2 and loss 2
 
        clip_logits=logits_adjustment(clip_logits,prob_map2.squeeze(0)[pred2].detach().cpu().item()-mean_loss[pred2].detach().cpu().item(),counter,count,alpha)
        output=clip_logits
        loss = softmax_entropy(clip_logits)
        prob_map = clip_logits.softmax(1)
        pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])
        
    #reflection 机制 判断是否相等
    if (pred == pred2) :
        sim=cos_sim(image_features,clip_weights.mT[pred])

        loss1=loss

    else:  
        doubt=True
        sim=cos_sim(clip_weights.mT[pred],clip_weights.mT[pred2])


        
        loss1=loss*torch.exp(-alpha1*sim) 

    return image_features, clip_logits, loss, prob_map, pred,doubt,image_features_all,loss1

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def get_clip_logits_3(image_features, clip_model, clip_weights,counter,count,alpha=1,mean_loss=None,alpha1=None):
    
  with torch.no_grad():

    if len(image_features.shape) == 1:
        image_features = image_features.unsqueeze(0)

    doubt=False
    clip_logits = 100. * image_features @ clip_weights

    if image_features.size(0) > 1:
        batch_entropy2 = softmax_entropy(clip_logits)
        selected_idx2 = torch.argsort(batch_entropy2, descending=False)[:int(batch_entropy2.size()[0] * 0.1)]
        output2 = clip_logits[selected_idx2]

        prob_map2 = output2.softmax(1).mean(0).unsqueeze(0)
        pred2 = int(output2.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        
        #caculate the pred2 and loss 2
        clip_logits=logits_adjustment(clip_logits,prob_map2.squeeze(0)[pred2].detach().cpu().item()-mean_loss[pred2].detach().cpu().item(),counter,count,alpha)
        batch_entropy = softmax_entropy(clip_logits)
        selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
        output = clip_logits[selected_idx]

        image_features = image_features[selected_idx].mean(0).unsqueeze(0)
        clip_logits = output.mean(0).unsqueeze(0)
        prob_map = output.softmax(1).mean(0).unsqueeze(0)
        loss = avg_entropy(output) 
        pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())  
    else:

        prob_map2 = clip_logits.softmax(1)
        pred2 = int(clip_logits.topk(1, 1, True, True)[1].t()[0])
        output2=clip_logits
        # caculate the pred2 and loss 2
 
        clip_logits=logits_adjustment(clip_logits,prob_map2.squeeze(0)[pred2].detach().cpu().item()-mean_loss[pred2].detach().cpu().item(),counter,count,alpha)
        output=clip_logits
        loss = softmax_entropy(clip_logits)
        prob_map = clip_logits.softmax(1)
        pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])


    return image_features, clip_logits, loss, prob_map, pred,doubt



    
import os
import yaml
import torch
import math
import numpy as np
import clip
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter
import torchvision.transforms as transforms
from PIL import Image
import json
from datetime import datetime
import open_clip
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import torch
import torch.nn as nn
import torch.nn.functional as F
class TextEncoderWithPrompt(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


def softmax_entropy(x):

    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc
def get_clip_logits(images, clip_model, clip_weights):
    # with torch.no_grad():
    if isinstance(images, list):
        images = torch.cat(images, dim=0).cuda()
    else:
        images = images.cuda()
    
    # Change 3D tensor to 4D tensor
    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    image_features = clip_model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    clip_logits = 100. * image_features @ clip_weights

    if image_features.size(0) > 1:
        batch_entropy = softmax_entropy(clip_logits)
        selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
        output = clip_logits[selected_idx]
        image_features = image_features[selected_idx].mean(0).unsqueeze(0)
        clip_logits = output.mean(0).unsqueeze(0)

        loss = avg_entropy(output)
        prob_map = output.softmax(1).mean(0).unsqueeze(0)
        pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
    else:
        loss = softmax_entropy(clip_logits)
        prob_map = clip_logits.softmax(1)
        pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

    return image_features, clip_logits, loss, prob_map, pred
def clip_classifier(classnames, template, cupl_path, clip_model, coop=False,setname=None, backbone='RN50'):

    f = open(cupl_path)
    cupl = json.load(f)
    
    if backbone == 'OpenCLIP':
        tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')

            texts = [t.format(classname) for t in template]
            if not (setname == 'eurosat' and backbone == 'ViT-B/16'):
              texts += cupl[classname]

            
            if coop:
                prompts = [f'a photo of a {classname}.']
                tokenized_prompts = clip.tokenize(prompts).cuda()
                embedding = clip_model.token_embedding(tokenized_prompts).type(clip_model.visual.conv1.weight.dtype)

                prefix = embedding[:, :1, :]
                suffix = embedding[:, 1 + n_ctx :, :]  # CLS, EOS
                
                # print(prefix.shape, ctx.shape, suffix.shape)

                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
                text_encoder_w_prompt = TextEncoderWithPrompt(clip_model)
                class_embedding = text_encoder_w_prompt(prompts, tokenized_prompts)
                class_embedding = class_embedding.squeeze()
            else:
                if backbone == 'RN50' or backbone == 'ViT-B/16':
                    texts = clip.tokenize(texts).cuda()
                elif backbone == 'OpenCLIP':
                    texts = tokenizer(texts).cuda()
                class_embeddings = clip_model.encode_text(texts)
                # prompt ensemble for ImageNet
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()           
    return clip_weights

def get_preprocess(is_augmix=True):
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                             std=[0.5, 0.5, 0.5]) # For OpenCLIP
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=is_augmix)

    return aug_preprocess

def get_preprocess2():
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                             std=[0.5, 0.5, 0.5]) # For OpenCLIP
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)

    return aug_preprocess

def get_config_file(config_path, dataset_name):
    if dataset_name == "I":
        config_name = "imagenet.yaml"
    elif dataset_name in ["A", "V", "R", "S"]:
        config_name = f"imagenet_{dataset_name.lower()}.yaml"
    else:
        config_name = f"{dataset_name}.yaml"
    
    config_file = os.path.join(config_path, config_name)
    
    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")

    return cfg


def build_test_data_loader(dataset_name, root_path, preprocess):
    if dataset_name == 'I':
        preprocess = get_preprocess(is_augmix=False)
        dataset = ImageNet(root_path, preprocess)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=1, num_workers=8, shuffle=True, pin_memory=True)
    
    elif dataset_name in ['A','V','S']:
        preprocess = get_preprocess(is_augmix=False)
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)
        
    elif dataset_name in ['R']:
        preprocess = get_preprocess()
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)
    
    

    elif dataset_name in ['eurosat','oxford_pets','oxford_flowers','caltech101']:
        #preprocess = get_preprocess()
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)
    
    elif dataset_name in ['dtd','fgvc','stanford_cars','sun397','food101']:
        preprocess= get_preprocess()
        dataset= build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)
    elif dataset_name in ['ucf101']:
        preprocess= get_preprocess(is_augmix=False)
        dataset= build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)
    
    else:
        raise "Dataset is not from the chosen list"
    
    return test_loader, dataset.classnames, dataset.template, dataset.cupl_path




