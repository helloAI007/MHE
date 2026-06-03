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

from skimage.transform import resize
import random
from typing import Union, Any, List
import xml.etree.ElementTree as ET
import open_clip
from PIL import Image
import timm

from utils import *

# Del and Ins
from evaluation import CausalMetric, auc, gkern


def handle_channels(img):
    if img.mode == 'L':  # Single channel image
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


################################ models ################################
# Wrapping class of unified model calling interface
class UnifiedModelWrapper(nn.Module):
    def __init__(self, model, preprocess, model_type="classification", class_names: List[str] = None):
        super(UnifiedModelWrapper, self).__init__() 
        self.model = model
        self.preprocess = preprocess
        self.model_type = model_type
        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # clip
        if self.model_type == "clip" and self.class_names is not None:
            self.text_tokens = open_clip.tokenize(self.class_names).to(self.device)
            with torch.no_grad():
                self.text_features = self.model.encode_text(self.text_tokens)
                self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        if self.model_type != "clip":
            self.model = nn.Sequential(model, nn.Softmax(dim=1))

    def __call__(self, image_input: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
            input_tensor = self.preprocess(image)
        elif isinstance(image_input, Image.Image):
            input_tensor = self.preprocess(image_input)
        elif isinstance(image_input, torch.Tensor):
            input_tensor = image_input
        else:
            raise ValueError("Unsupported input type")
        
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            if self.model_type == "clip":
                image_features = self.model.encode_image(input_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity (returned as logits)
                logits = (100.0 * image_features @ self.text_features.T)
                logits[0] = torch.nn.functional.softmax(logits[0], dim=0)
                return logits
            else:
                # other
                return self.model(input_tensor)
    
    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        if hasattr(self, 'text_features'):
            self.text_features = self.text_features.to(device)
        return self
# Standard preprocessing function
standard_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model_loaders = {
    "ResNet": lambda: models.resnet50(pretrained=True),
    "VGG16": lambda: models.vgg16(pretrained=True),
    "ViT": lambda: timm.create_model('vit_base_patch16_224', pretrained=True),
    "DINO": lambda: timm.create_model('vit_base_patch16_224_dino', pretrained=True),
}

################################ metric ################################

# Used for single category images
def get_all_boxes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append([xmin, ymin, xmax, ymax])
    return bboxes
    
    
def DelAndInsFunc(model, input_, sal, verboseDel=1, verboseIns=1, save_to=None):
    cudnn.benchmark = True
    klen = 11
    ksig = 5
    kern = gkern(klen, ksig)

    # Function that blurs input image
    blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2) 

    if type(sal) == torch.Tensor:
        sal = sal.cpu().numpy() 
    input_img = input_.cpu().detach()
    
    # DELETION
    full_like = lambda x: torch.full_like(x, x.min()) 
    deletion = CausalMetric(model, 'del', 224, substrate_fn=full_like) 
    h = deletion.single_run(input_img, sal, verbose=verboseDel, save_to=save_to)
    delete = auc(h)
    
    # INSERTION
    insertion = CausalMetric(model, 'ins', 224, substrate_fn=blur)
    h = insertion.single_run(input_img, sal, verbose=verboseIns, save_to=save_to)
    insert = auc(h)
    
    cudnn.benchmark = False
    return delete, insert

def AD_AI_once(
    model: torch.nn.Module,
    input_image: torch.Tensor,
    cam_map: torch.Tensor,
    class_idx: Union[int, None] = None
) -> tuple:
    probs = model(input_image).data
    _, indexs = torch.topk(probs, 2)
    top1_index = indexs[0][0] # top1.
    probs = probs[0][top1_index]

    masked_input = cam_map * input_image
    masked_probs = model(masked_input).data
    masked_probs = masked_probs[0][top1_index]

    # drop
    drop = torch.relu(probs - masked_probs).div(probs + 1e-7)

    # increase
    increase = probs < masked_probs

    return drop.sum().item(), increase.sum().item()

def pointing_game(saliency_map, bbox):
    """
    Calculate the Pointing Game score for a single or multiple bounding boxes.
    Parameters:
        - saliency_map: 2D-array in range [0, 1]
        - bbox: Single bounding box (torch.Size([7])) or multiple (torch.Size([N, 7])).
    Returns: PG scores for the saliency map.
    """

    if len(bbox.shape) == 1:  # (np.array([7]))
        bbox = bbox[np.newaxis, :]  # to np.array([1, 7])

    scores = []
    for box in bbox:  # Iterate over all bounding boxes
        x_min, y_min, x_max, y_max = box[:4]
        x_min, y_min, x_max, y_max = map(lambda x: max(int(x), 0), [x_min, y_min, x_max, y_max])

        # Find the maximum saliency point
        max_saliency = np.max(saliency_map)
        point_y, point_x = np.where(saliency_map == max_saliency)

        # Check for hits and misses
        hit = 0
        miss = 0
        for px, py in zip(point_x, point_y):
            if x_min <= px < x_max and y_min <= py < y_max:
                hit += 1
            else:
                miss += 1

        # Calculate PG score
        total = hit + miss
        score = hit / total if total > 0 else 0
        scores.append(score)

    return scores if len(scores) > 1 else scores[0]

def energy_based_pointing_game(saliency_map, bbox):
    """
    Calculate the EBPG score for a single or multiple bounding boxes.
    Parameters:
        - saliency_map: 2D-array in range [0, 1]
        - bbox: Single bounding box (torch.Size([7])) or multiple (torch.Size([N, 7]))
    Returns: EBPG scores for the saliency map.
    """
    if len(bbox.shape) == 1:  # Single bounding box (torch.Size([7]))
        bbox = bbox.unsqueeze(0)  # Convert to torch.Size([1, 7])

    scores = []
    for box in bbox:  # Iterate over all bounding boxes -> consider all detected objects in the image
        x_min, y_min, x_max, y_max = box[:4]
        x_min, y_min, x_max, y_max = map(lambda x: max(int(x), 0), [x_min, y_min, x_max, y_max])

        # Create bounding box mask
        mask = np.zeros_like(saliency_map)
        mask[y_min:y_max, x_min:x_max] = 1 # y=rows, x=columns

        # Normalize saliency map if needed
        if saliency_map.max() > 1.0:
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

        # Calculate energy
        energy_bbox = np.sum(saliency_map * mask)
        energy_whole = np.sum(saliency_map)

        # Calculate EBPG score
        score = energy_bbox / energy_whole if energy_whole > 0 else 0
        scores.append(score)

    return scores if len(scores) > 1 else scores[0]

################################ helper function ################################

def check_path_exist(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"directory {directory_path} has been created.")
    else:
        print(f"directory {directory_path} already exists.")

def oneResize(one_mask, img_size):
    mask = resize(one_mask, img_size, order=1, mode='edge', anti_aliasing=False)
    return mask

def update_matrix_ByProb(matrix, positions, probability=0.5):
    if len(positions) == 0 : 
        # If the position is empty, randomly generate one
        Last_Mask = matrix
        random_matrix = np.random.rand(Last_Mask.shape[0], Last_Mask.shape[1] )
        matrix = np.zeros((Last_Mask.shape[0], Last_Mask.shape[1]))
        matrix[0.5 > random_matrix] = 1
        return matrix

    positions_array = np.array(positions) 
    random_values = np.random.rand(len(positions[0])) < probability 
    matrix[positions_array[0,:], positions_array[1,:]]  = random_values.astype(int)

    if np.all(matrix == 0.0): # Prevent the generated mask from being empty
        Last_Mask = matrix
        random_matrix = np.random.rand(Last_Mask.shape[0], Last_Mask.shape[1] )
        matrix = np.zeros((Last_Mask.shape[0], Last_Mask.shape[1]))
        matrix[0.5 > random_matrix] = 1

    return matrix

# Return a random mask of the same size as the Last_Mask
def random_mask(Last_Mask): 
    random_matrix = np.random.rand(Last_Mask.shape[0], Last_Mask.shape[1] )
    matrix = np.zeros((Last_Mask.shape[0], Last_Mask.shape[1]))
    matrix[0.5 > random_matrix] = 1
    return matrix

# Generate a mask randomly based on the probability from Matrix.
def random_mask_From(fromMatrix): 
    random_matrix = np.random.rand(fromMatrix.shape[0], fromMatrix.shape[1] )
    matrix = np.zeros((fromMatrix.shape[0], fromMatrix.shape[1]))
    matrix[fromMatrix > random_matrix] = 1
    return matrix

def is_all_ones(matrix):
    np_matrix = np.array(matrix)
    return np.all(np_matrix == 1)

def generate_mask(input_matrix, Score_dist, Last_Mask, IsSuggestion):
    belta = 0.01 # 0.5

    if is_all_ones(input_matrix):
        input_matrix = random_mask(input_matrix)

    if IsSuggestion:
        new_mask = random_mask_From(Last_Mask)
    else:
        # Weaken the illuminated area. The larger the gap, the greater the probability of flipping
        new_mask = update_matrix_ByProb(input_matrix, np.where(input_matrix==1), probability=(1-np.abs(Score_dist))-belta )
        # Random walk increases negative areas (other areas).
        new_mask = update_matrix_ByProb(new_mask, np.where(input_matrix==0), probability=(np.abs(Score_dist))+belta )
    
    if is_all_ones(new_mask):
        new_mask = random_mask(new_mask)
    return new_mask

def distance_L4(a, b):
    return (np.sum((a - b)**4))**(1/4)

def get_predict_class(model,input_image, topk=5):
    logit = model(input_image).data 
    value, indexs = torch.topk(logit, topk)
    return value[0], indexs[0], logit[0]

def calculate_probability(i, j, d):
    """
    Calculate the probability from matrix i to matrix j.
    d: "score difference" of matrix i.
    
    Args:
        i (numpy.ndarray): Initial 0-1 binary matrix.
        j (numpy.ndarray): Target 0-1 binary matrix.
        d (float): Probability value for constructing i's probability matrix.
        
    Returns:
        float: Final probability value.
    """
    if i.shape != j.shape:
        raise ValueError("The size of matrices i and j must be the same")
    
    probability_matrix = np.full(i.shape, 1-d-0.01)
    
    # Count the positions that need to be changed from i to j
    change_matrix = (i != j)
    
    probability_matrix[change_matrix] = d+0.01
    
    final_probability = np.prod(probability_matrix)
    
    return final_probability
def GetAdviceRate(acc_unresized_mask_n, acc_score_dist, one_mask, score_dist):
    # old -> cur
    forward_rate = calculate_probability(acc_unresized_mask_n, one_mask, acc_score_dist)
    # cur -> old
    back_rete = calculate_probability(one_mask, acc_unresized_mask_n, score_dist)
    return forward_rate,back_rete

def get_Dist_PNN(topktoExplain, origin_TopK_score, target_TopK_class_index, saliency_resized_topk, input_image, model, control):
    test_score = np.zeros(topktoExplain)

    P_dec = np.zeros(topktoExplain)
    N_inc = np.zeros(topktoExplain)
    Neutral = np.zeros(topktoExplain)
    PNN = np.zeros(topktoExplain) 

    # save_to = './'+control['save_path']+'/'+control['img_name'] +"/"
    # a = 0.5
    # b = 0.1
    a = control["abla_a"]
    b = control["abla_b"]

    for i in range(topktoExplain):
        origin_score = origin_TopK_score[i]
        saliency_resized = saliency_resized_topk[i]
        target_class_index_i = target_TopK_class_index[i]

        test_img = input_image * normalization_mask(saliency_resized)
        test_img1 = ( test_img ).float()
        logit = model(test_img1).cuda()
        score = logit[0][target_class_index_i].squeeze() 
        test_score[i] = score

        saliency_map = normalization_mask(saliency_resized) # Normalize

        mask_high = saliency_map.copy()
        mask_high[mask_high >= a] = 0
        test_img_high = input_image * mask_high
        test_img_high = test_img_high.float()
        logit_high = model(test_img_high).cuda()
        score_high = logit_high[0][target_class_index_i].squeeze()
        P_dec[i] = origin_score.cpu().numpy() - score_high.cpu().numpy()
    
        mask_mid = saliency_map.copy()
        mask_mid[(mask_mid >= b) & (mask_mid < a)] = 0
        test_img_mid = input_image * mask_mid
        test_img_mid = test_img_mid.float()
        logit_mid = model(test_img_mid).cuda()
        score_mid = logit_mid[0][target_class_index_i].squeeze()
        N_inc[i] = score_mid.cpu().numpy() - origin_score.cpu().numpy()

        mask_low = saliency_map.copy()
        mask_low[mask_low < b] = 0
        test_img_low = input_image * mask_low
        test_img_low = test_img_low.float()
        logit_low = model(test_img_low).cuda()
        score_low = logit_low[0][target_class_index_i].squeeze()
        Neutral[i] = abs(origin_score.cpu().numpy() - score_low.cpu().numpy())

        PNN[i] = (P_dec[i] + N_inc[i] - Neutral[i] + 3.0)/5.0

    # 0: The absolute value of difference
    # 1: explanation score
    # 2: original score.
    result_Score_Dist = [abs(test_score[0] - origin_TopK_score[0].cpu().numpy()), test_score, origin_TopK_score.cpu().numpy()] # 
    
    # top-1:
    Result_Score_PNN = [PNN[0], P_dec[0], N_inc[0], Neutral[0]]
    return [result_Score_Dist, Result_Score_PNN] 

def normalization_mask(cur_mask):
    min_val = np.min(cur_mask)
    max_val = np.max(cur_mask)
    one_mask = ( cur_mask-min_val )/( max_val - min_val )
    return one_mask

################################ base function ################################

def MHEFunc(model, control, img_name, times, target_class_index, kernel_size=(10, 10)):
    threshhold_ablation = control["threshhold_ablation"] 
    times = control["terms_ablation"] 
    size_ablation = control["size_ablation"] 
    kernel_size = (size_ablation, size_ablation)
    distance = distance_L4

    topktoExplain = control["topktoExplain"]
    target_TopK_class_index = control["target_TopK_class_index"]
    origin_TopK_score = control["origin_TopK_score"]
    origin_score_vector = control['origin_score']
    origin_score = origin_score_vector[target_class_index]

    skip_flag = control["skip_flag"] 

    # only for MHE-pro
    Last_Mask = control["Last_Mask"]
    IsSuggestion = control["IsSuggestion"] # The first MHE is random, followed by the use of prior knowledge
    
    # Input image
    input_image = read_tensor(img_name).data
    img_size = (224,224)

    # information record
    is_accept = False
    accepted_masks = []
    accepted_unresized_masks = [] # Unstretched mask for MHE-pro
    accepted_scores = []
    
    # Skip the first k=200 rounds
    skip_times = 200

    # initialization
    acc_unresized_mask = np.random.rand(kernel_size[0], kernel_size[1] )
    acc_unresized_mask[acc_unresized_mask <0.5] = 0
    acc_unresized_mask[acc_unresized_mask > 0.5] = 1
    acc_unresized_mask = normalization_mask(acc_unresized_mask)
    acc_mask = oneResize(acc_unresized_mask, img_size) 
    acc_mask = normalization_mask(acc_mask) 
    # Initialize acc_score
    input_image1 = (input_image * acc_mask).float()
    logit = model(input_image1).cuda()
    acc_score = logit[0].cpu().numpy()
    score = acc_score
    score_dist = distance(score, origin_score_vector.cpu().numpy())
    acc_score_dist = score_dist 

    for i in range(times):
        # Generate test mask samples
        acc_unresized_mask_n = normalization_mask(acc_unresized_mask)
        cur_mask = generate_mask(acc_unresized_mask_n, acc_score_dist, Last_Mask, IsSuggestion)

        min_val = np.min(cur_mask)
        max_val = np.max(cur_mask)
        one_mask = ( cur_mask-min_val )/( max_val - min_val )
        mask = oneResize(one_mask, img_size)  
        mask = normalization_mask(mask)
        # testing
        input_image1 = (input_image * mask).float()
        logit = model(input_image1).cuda()
        score = logit[0]
        predicted_class = logit.max(1)[-1]
            
        # Calculate acceptance rate alpha.
        score_dist = distance(score.cpu().numpy(), origin_score_vector.cpu().numpy())
        acc_2_i, i_2_acc = GetAdviceRate(acc_unresized_mask_n, acc_score_dist, one_mask, score_dist)
        alpha_top = np.exp(- score_dist )*i_2_acc
        alpha_buttom = np.exp(- acc_score_dist )*acc_2_i
        alpha = alpha_top / alpha_buttom
        
        # acception of MHE
        if np.abs(score_dist) <= threshhold_ablation:
            is_accept = True
            if score_dist <= (threshhold_ablation/10) :
                area_one = np.linalg.norm(one_mask)
                area_acc = np.linalg.norm(acc_unresized_mask)
                # Prioritize accepting smaller areas
                if area_acc >= area_one : 
                    acc_mask = mask 
                    acc_score = score.cpu().numpy()
                    acc_unresized_mask = one_mask
                    acc_score_dist = score_dist
                else: # Keep the original acceptance unchanged
                    acc_score_dist = score_dist
            else:
                acc_mask = mask 
                acc_score = score.cpu().numpy()
                acc_unresized_mask = one_mask
                acc_score_dist = score_dist
        # The follow-up is only a transition and will not be used for calculating explanatory diagrams
        elif alpha > 1:
            acc_mask = mask
            acc_score = score.cpu().numpy()
            acc_unresized_mask = one_mask
            acc_score_dist = score_dist        
        elif alpha == 1 : 
            alpha_top = np.exp(-(np.abs(one_mask).sum() ))
            alpha_buttom = np.exp(-(np.abs(acc_unresized_mask).sum() ))
            alpha_area = alpha_top / alpha_buttom
            if alpha_area >= 1: 
                acc_mask = mask 
                acc_score = score.cpu().numpy()
                acc_unresized_mask = one_mask
                acc_score_dist = score_dist
        elif alpha < 1 and alpha > np.random.rand(): # random
            acc_mask = mask 
            acc_score = score.cpu().numpy()
            acc_unresized_mask = one_mask 
            acc_score_dist = score_dist
        else: # refuse
            is_accept = False

        # Not accepted during warm-up phase
        if skip_flag and i < skip_times:
            continue
        elif skip_flag and i >= skip_times:
            skip_flag = False
        
        # Record mask for settlement results
        if acc_mask is not None and is_accept and acc_mask.shape:
            accepted_unresized_masks.append(acc_unresized_mask) 
            accepted_masks.append(acc_mask)
            if topktoExplain == 1:
                accepted_scores.append(acc_score[target_class_index])
            else:
                accepted_scores.append(acc_score )
            is_accept = False # recover

    accepted_masks = np.array(accepted_masks) 
    accepted_scores = np.array(accepted_scores)
    accepted_unresized_masks = np.array(accepted_unresized_masks) 

    # Calculation Explanation maps
    saliency_resized_topk = []
    for class_index in target_TopK_class_index:
        weight_mask = accepted_masks * accepted_scores[:, class_index, None, None] 
        saliency_resized = np.sum(weight_mask, axis=0)/(len(accepted_masks) *1.0) 
        saliency_resized_topk.append(saliency_resized)
    weight_unresized_mask = accepted_unresized_masks # only for MHE-pro
    unresized_masks = np.sum(weight_unresized_mask, axis=0)/(len(accepted_unresized_masks) *1.0) 

    # save map
    if control["saliency"] and saliency_resized.shape:
        check_path_exist('./'+control['save_path']+'/'+control['img_name'])
        test_score = np.zeros(topktoExplain)
        for i in range(topktoExplain): 
            origin_score = origin_TopK_score[i]
            saliency_resized = saliency_resized_topk[i]
            target_class_index_i = target_TopK_class_index[i]
            
            # Calculate the classification confidence of the explanation
            test_img = input_image * normalization_mask(saliency_resized)
            test_img1 = ( test_img ).float()
            logit = model(test_img1).cuda()
            score = logit[0][target_class_index_i].squeeze() 

            # save
            tensor_imshow(input_image[0])
            plt.imshow(saliency_resized, cmap='jet', interpolation='nearest', alpha=0.5)
            # plt.show()
            plt.savefig('./'+control['save_path']+'/'+control['img_name']\
            +f'/saliency_top{i}_{control["terms"]}_test={float(score):.5f}_origin={float(origin_score.cpu().numpy()):.5f}.png', dpi=300, bbox_inches='tight')
            plt.close()

            test_score[i] = score

    result_Score_2 = get_Dist_PNN(topktoExplain, origin_TopK_score, target_TopK_class_index, saliency_resized_topk, input_image,model, control)

    return saliency_resized_topk, unresized_masks, result_Score_2

################################ MHE & MHE-pro ################################

# Generate mask size sequence for MHE-pro
def get_auto_size(start, end, step): 
    lst = list(range(start, end, step))  
    if lst[-1] + step > end and lst[-1] != end:  
        lst.append(end)
    return lst

def mhe_pro(model, control, img_name, target_class_index, times=1000, kernel_size=(8,8)):
    startsize = 5 
    endsize = 10 
    sizeStep = 2
    ksizes = get_auto_size(startsize, endsize, sizeStep)
    terms = [1000] * len(ksizes) # Each round is 1000

    control["skip_flag"] = True # Do you want to skip the first N rounds ?
    control["IsSuggestion"] = False # False for first Size/phase/MHE module.

    Last_Mask = control["Last_Mask"] = np.full(kernel_size, 0.5 ) # Initialization (not used in the first attempt)

    for i in range(0,len(ksizes)):
        control["terms"] = i # phase tag 

        kernel_size=(ksizes[i], ksizes[i])
        control["size_ablation"] = ksizes[i]
        one_term = terms[i]
        control["terms_ablation"] = one_term

        # Stretch to the next kernel size
        Last_Mask = resize(Last_Mask, kernel_size, order=1, mode='reflect', anti_aliasing=False) 
        control["Last_Mask"] = normalization_mask(Last_Mask)

        # Complete a single stage
        saliency_resized_topk, Last_Mask,  result_Score_Dist = MHEFunc(model, control, img_name, one_term, target_class_index, kernel_size)

        if i == 0: # Subsequent stages after the first round
            control["skip_flag"] = False # Warm up is over
            control["IsSuggestion"] = True # Use prior information from the previous module

        if np.any(np.isnan(Last_Mask)):
            # ValueError: probabilities contain NaN
            return -1,-1
    return saliency_resized_topk, result_Score_Dist

def mhe(model, control, img_name, target_class_index, times=4000, kernel_size=(8,8)):
    control["terms"] = 0 # phase tag 
    control["skip_flag"] = True 
    control["IsSuggestion"] = False

    kernel_size=(control["size_ablation"], control["size_ablation"])
    control["Last_Mask"] = np.full(kernel_size, 1.0/(kernel_size[0]*kernel_size[1]))

    saliency_resized_topk, _, result_Score_Dist = MHEFunc(model, control, img_name, 4000, target_class_index, kernel_size)

    return saliency_resized_topk, result_Score_Dist

################################ MHE-e ################################

def mhe_e(model, control, img_name, target_class_index, times=4000, kernel_size=(8,8)):
    control["terms"] = 0 
    kernel_size=(control["size_ablation"], control["size_ablation"])
    control["skip_flag"] = True 
    control["IsSuggestion"] = False

    control["Last_Mask"] = np.full(kernel_size, 1.0/(kernel_size[0]*kernel_size[1])) # 归一化
    
    saliency_resized_topk, _, result_Score_Dist = MHE_e_Func(model, control, img_name, 4000, target_class_index, kernel_size)
    
    return saliency_resized_topk, result_Score_Dist


def MHE_e_Func(model, control, img_name, times, target_class_index, kernel_size=(10, 10)):
    threshhold_ablation = control["threshhold_ablation"] 
    times = control["terms_ablation"] 
    size_ablation = control["size_ablation"] 
    kernel_size = (size_ablation, size_ablation)
    distance = distance_L4

    topktoExplain = control["topktoExplain"]
    target_TopK_class_index = control["target_TopK_class_index"]
    origin_TopK_score = control["origin_TopK_score"]
    origin_score_vector = control['origin_score']
    origin_score = origin_score_vector[target_class_index]

    skip_flag = control["skip_flag"] 

    # Input image
    input_image = read_tensor(img_name).data
    img_size = (224,224)

    # information record
    is_accept = False
    accepted_masks = []
    accepted_scores = []
    
    # Skip the first k=200 rounds
    skip_times = 200

    # initialization
    acc_unresized_mask = np.random.rand(kernel_size[0], kernel_size[1] )
    acc_unresized_mask[acc_unresized_mask <0.5] = 0
    acc_unresized_mask[acc_unresized_mask > 0.5] = 1
    acc_unresized_mask = normalization_mask(acc_unresized_mask)
    acc_mask = oneResize(acc_unresized_mask, img_size) 
    acc_mask = normalization_mask(acc_mask) 
    # Initialize acc_score
    input_image1 = (input_image * acc_mask).float()
    logit = model(input_image1).cuda()
    acc_score = logit[0].cpu().numpy()
    score = acc_score
    score_dist = distance(score, origin_score_vector.cpu().numpy())
    acc_score_dist = score_dist 

    for i in range(times):
        acc_unresized_mask_n = normalization_mask(acc_unresized_mask)
        cur_mask = generate_mask(acc_unresized_mask_n, acc_score_dist, None, False)

        one_mask = cur_mask
        min_val = np.min(cur_mask)
        max_val = np.max(cur_mask)
        one_mask = ( cur_mask-min_val )/( max_val - min_val )
        mask = oneResize(one_mask, img_size)      
        mask = normalization_mask(mask) 
        
        input_image1 = (input_image * mask).float()
        logit = model(input_image1).cuda()
        score = logit[0]
        predicted_class = logit.max(1)[-1] 
        
        # MHE-e use confidence score rather than distance
        score_dist = 1- score[target_class_index].cpu().numpy() 
        score_dist = max(min(score_dist,0.1), 0.9)

        alpha_top = score[target_class_index].cpu().numpy() 
        alpha_buttom = origin_score_vector[target_class_index].cpu().numpy()
        alpha = alpha_top / alpha_buttom
        
        if alpha > 1:
            is_accept = True
            acc_mask = mask 
            acc_score = score.cpu().numpy()
            acc_unresized_mask = one_mask
            acc_score_dist = score_dist        
        elif alpha == 1 : 
            is_accept = True
            alpha_top = np.exp(-(np.abs(one_mask).sum() ))
            alpha_buttom = np.exp(-(np.abs(acc_unresized_mask).sum() ))
            alpha_area = alpha_top / alpha_buttom
            if alpha_area >= 1: 
                acc_mask = mask 
                acc_score = score.cpu().numpy()
                acc_unresized_mask = one_mask
                acc_score_dist = score_dist
        elif alpha < 1 and alpha > np.random.rand(): # random
            acc_mask = mask 
            acc_score = score.cpu().numpy()
            acc_unresized_mask = one_mask 
            acc_score_dist = score_dist
        else: 
            is_accept = False

        # warm-up
        if skip_flag and i < skip_times:
            continue
        elif skip_flag and i >= skip_times:
            skip_flag = False

        if acc_mask is not None and is_accept and acc_mask.shape: 
            accepted_masks.append(acc_mask )
            if topktoExplain == 1:
                accepted_scores.append(acc_score[target_class_index] )
            else:
                accepted_scores.append(acc_score ) 
            is_accept = False

    accepted_masks = np.array(accepted_masks) 
    accepted_scores = np.array(accepted_scores)

    saliency_resized_topk = []
    for class_index in target_TopK_class_index:
        weight_mask = accepted_masks * accepted_scores[:, class_index, None, None] 
        saliency_resized = np.sum(weight_mask, axis=0)/(len(accepted_masks) *1.0) 
        saliency_resized_topk.append(saliency_resized)

    # save 
    if control["saliency"] and saliency_resized.shape: 
        check_path_exist('./'+control['save_path']+'/'+control['img_name'])
        test_score = np.zeros(topktoExplain)
        for i in range(topktoExplain): 
            origin_score = origin_TopK_score[i]
            saliency_resized = saliency_resized_topk[i]
            target_class_index_i = target_TopK_class_index[i]
            
            test_img = input_image * normalization_mask(saliency_resized)
            test_img1 = ( test_img ).float()
            logit = model(test_img1).cuda()
            score = logit[0][target_class_index_i].squeeze() 
            
            tensor_imshow(input_image[0])
            plt.imshow(saliency_resized, cmap='jet', interpolation='nearest', alpha=0.5)
            # plt.show()
            plt.savefig('./'+control['save_path']+'/'+control['img_name']\
            +f'/saliency_top{i}_{control["terms"]}_test={float(score):.5f}_origin={float(origin_score.cpu().numpy()):.5f}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            test_score[i] = score

    result_Score_2 = get_Dist_PNN(topktoExplain, origin_TopK_score, target_TopK_class_index, saliency_resized_topk, input_image,model, control)

    return saliency_resized_topk, None, result_Score_2
