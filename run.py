import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from utils import *

from skimage.transform import resize
import random
from PIL import Image
import sys
import timm
import open_clip

from mhe import mhe_pro,mhe,mhe_e,\
get_predict_class,DelAndInsFunc,check_path_exist, AD_AI_once, normalization_mask, \
pointing_game, energy_based_pointing_game, get_all_boxes

# from mhe import model_names
from mhe import model_loaders,UnifiedModelWrapper

# your input images directory
imgs_dir = './imgs/'
# your directory where the bounding box annotation file is located
# Only set annotations_dir and is_PG when you want to test PG/EBPG
annotations_dir = "xx_xml/" # only for PG/EBPG
is_PG = False # Whether to evaluate PG, EBPG.

control = dict()
control["saliency"] = True # Do you want to save the image of the explanation result?
control["threshhold_ablation"] = 0.1 # Score acceptance threshold: 0.1, 0.05, 0.2, 0.3
control["terms_ablation"] = 4000 # rounds: 1000, 2000, 3000, 4000, 5000, 6000
control["size_ablation"] = 8 # mask size: 6x6, 8x8, 10x10
control["abla_a"] = 0.5 # threshold a in PNN
control["abla_b"] = 0.1 # threshold b in PNN


image_paths = [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) if f.endswith(('.png', '.jpg', '.jpeg','.JPEG'))]

perform_number = sys.argv[1]

def Check_xml(xml_path):
    if  not os.path.exists(xml_path):
        print(f"{xml_path} xml not found")
    return xml_path
#

def handle_channels(img):
    if img.mode == 'L': 
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    return img
read_tensor = transforms.Compose([
    lambda x: Image.open(x),
    handle_channels,  
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    lambda x: torch.unsqueeze(x, 0)
])

model_names = [
    "ResNet","VGG16", "ViT", "DINO","CLIP"
]

# Set save path
if perform_number == "1":
    control['save_path'] = "results/mhe"
    black_explain_perform = mhe
elif perform_number == "2":
    control['save_path'] = "results/mhe_pro" 
    black_explain_perform = mhe_pro 
elif perform_number == "3":
    control['save_path'] = "results/mhe_e"
    black_explain_perform = mhe_e

def RunExp(image_paths, black_explain_perform, model, control):
    # record
    all_delete = []
    all_insert = []

    # AD, AI
    all_drop = [] 
    all_increase = [] 

    all_pg = []
    all_ebpg = []

    all_dist = [] # dis
    all_PNN_result = [] # PNN

    for image_path in image_paths:
        control['img_name']= os.path.splitext(os.path.basename(image_path))[0]
        if is_PG: # for PG,EBPG.
            xml_path = annotations_dir + control['img_name']+".xml" # .xml annotation file
            if not os.path.exists(xml_path):
                print(f"{xml_path} not found")
                continue
        
        input_image = read_tensor(image_path).data
        
        # Explain the topK target
        topK = 4 
        control["topktoExplain"] = topK 
        origin_TopK_score, target_TopK_class_index, origin_score = get_predict_class(model,input_image, topK) 
        control["target_TopK_class_index"] = target_TopK_class_index
        control["origin_TopK_score"] = origin_TopK_score
        control['origin_score'] = origin_score 
        target_class_index = target_TopK_class_index[0]

        if is_PG:
            bboxes = np.array(get_all_boxes(xml_path))
            if len(bboxes) <= 0:
                print("The category box does not exist, imgName=",control['img_name']," className=",get_class_name(target_TopK_class_index[0]))
                continue

        # Preventing a possible search failure in MH's Markov chain
        for attempt in range(3):
            saliency_resized_topk, result_Score_2 = black_explain_perform(model, control, image_path, target_class_index)
            result_Score_Dist = result_Score_2[0]
            Result_Score_PNN = result_Score_2[1]

            if saliency_resized_topk is None:
                print(f"sampling retrying...")
                continue
            else:
                break
        if saliency_resized_topk is None:
            print("retrying failed")
            continue
        
        # input_image = read_tensor(image_path).data 
        delete, insert = DelAndInsFunc(model, input_image, saliency_resized_topk[0], 1, 1, None)

        drop, increase = AD_AI_once(model, input_image, torch.from_numpy(normalization_mask(saliency_resized_topk[0])).float() )
        
        if is_PG:
            pg = pointing_game(saliency_resized_topk[0], bboxes)
            pg = np.mean(np.array(pg))
            ebpg = energy_based_pointing_game(saliency_resized_topk[0], bboxes)
            ebpg = np.mean(np.array(ebpg))
            
            all_pg.append(pg)
            all_ebpg.append(ebpg)
        
        all_delete.append(delete)
        all_insert.append(insert)

        all_drop.append(drop)
        all_increase.append(increase)

        all_dist.append(result_Score_Dist[0])
        all_PNN_result.append(Result_Score_PNN)

    # Record metrics
    all_delete_np = np.array(all_delete)
    all_insert_np = np.array(all_insert)
    all_delete_np = all_delete_np[~np.isnan(all_delete_np)] # Remove potential Nan
    all_insert_np = all_insert_np[~np.isnan(all_insert_np)]

    all_dist_np = np.array(all_dist)
    all_dist_np_NoNan = all_dist_np[~np.isnan(all_dist_np)]

    all_PNN_result_np = np.array(all_PNN_result)
    all_PNN_result_np = all_PNN_result_np[~np.isnan(all_PNN_result_np).any(axis=1)]
    all_PNN_result_np_mean = np.mean(all_PNN_result_np, axis=0)

    all_drop_np = np.array(all_drop)
    all_drop_np = all_drop_np[~np.isnan(all_drop_np)] 
    all_increase_np = np.array(all_increase)
    all_increase_np = all_increase_np[~np.isnan(all_increase_np)]

    text_lines = [
        f"count of images ={len(image_paths)}",
        f"dis: mean={np.mean(all_dist_np_NoNan)}",
        f"PNN={all_PNN_result_np_mean[0]}, P_dec={all_PNN_result_np_mean[1]}, N_inc={all_PNN_result_np_mean[2]}, Neutral={all_PNN_result_np_mean[3]}",
        f"Delete: mean={np.mean(all_delete_np)}, max={np.max(all_delete_np)}, min={np.min(all_delete_np)}.",
        f"Insert: mean={np.mean(all_insert_np)}, max={np.max(all_insert_np)}, min={np.min(all_insert_np)}.",
        f"AD: mean={np.mean(all_drop_np)}, max={np.max(all_drop_np)}, min={np.min(all_drop_np)}.",
        f"AI: mean={np.mean(all_increase_np)}, max={np.max(all_increase_np)}, min={np.min(all_increase_np)}.",
    ] 

    if is_PG:
        all_pg_np = np.array(all_pg)
        all_ebpg_np = np.array(all_ebpg)
        all_pg_np = all_pg_np[~np.isnan(all_pg_np)]
        all_ebpg_np = all_ebpg_np[~np.isnan(all_ebpg_np)]
        text_lines.append(f"PG: mean={np.mean(all_pg_np)}, max={np.max(all_pg_np)}, min={np.min(all_pg_np)}.")
        text_lines.append(f"EBPG: mean={np.mean(all_ebpg_np)}, max={np.max(all_ebpg_np)}, min={np.min(all_ebpg_np)}.")

    check_path_exist('./'+ control['save_path'])
    file_path = './'+ control['save_path'] + '/result.txt'
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in text_lines:
            file.write(line + '\n')


root_path = control['save_path']

# CLIP
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model_loaders["CLIP"] = lambda: clip_model
preprocess_mapping = {
    "CLIP": clip_preprocess
}
for name in model_names:
    if name not in preprocess_mapping:
        preprocess_mapping[name] = read_tensor
model_type_mapping = {
    "CLIP": "clip"
}
for name in model_names:
    if name not in model_type_mapping:
        model_type_mapping[name] = "classification"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_class_names = ["a dog", "a cat", "a bird", "a car", "a person", "a tree"]
wrapped_models = []
for name in model_names:
    model = model_loaders[name]()
    preprocess = preprocess_mapping[name]
    model_type = model_type_mapping[name]
    
    class_names = clip_class_names if name == "CLIP" else None
    
    wrapped_model = UnifiedModelWrapper(
        model.to(device), preprocess, model_type, class_names
    )
    
    wrapped_models.append({
        "name": name,
        "model": wrapped_model
    })


for model_info in wrapped_models:
    try:
        model = model_info["model"]
        name = model_info["name"]
    except Exception as e: 
        print(f"load {name} err: {e}")
        sys.exit(1)  

    model = model.eval()
    model = model.cuda()
    for p in model.parameters():
        p.requires_grad = False

    print(f"{name} loaded")

    control['save_path'] = root_path +'/'+name
    with torch.no_grad():
        RunExp(image_paths, black_explain_perform, model, control)

