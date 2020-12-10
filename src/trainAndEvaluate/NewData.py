import random, string
import numpy as np


def generate_acid_color_proportions():
    min_prop=10
    sum_prop = 100
    black_threshold = 40
    white_threshold=40
    possible_props = list(range(0,100))
    choosen_props = []
    acid_color_prop_maps = {}
    for i in range(len(string.ascii_uppercase)):
        flag=True
        acid = string.ascii_uppercase[i]
        while flag:
            first_prop = random.choice(possible_props)
            second_prop = random.choice(possible_props)
            third_prop = random.choice(possible_props)
            if first_prop +  second_prop + third_prop == sum_prop and (first_prop, second_prop, third_prop) not in choosen_props and min(first_prop, second_prop, third_prop)>=min_prop and first_prop<white_threshold and third_prop<black_threshold:
            
                flag=False
                assert (first_prop, second_prop, third_prop) not in choosen_props
                choosen_props.append((first_prop, second_prop, third_prop))
                acid_color_prop_maps[acid] = [first_prop, second_prop, third_prop]

    return acid_color_prop_maps


def sequenc2histogram(seq):
    """
    Converts seq to image inthe form of hist
    """
    acid_color_prop_maps = generate_acid_color_proportions()
    acid_width = 1
    img_h = 150
    img_w = len(seq)*acid_width
    img = np.zeros((img_h, img_w, 3))
    for index,acid in enumerate(seq):
        color_props = acid_color_prop_maps[acid]
        white = (255,255,255)
        black = (0,0,0)
        gray = (90,90,90)
        first_split = round(int(img_h*color_props[0]*0.01))
        second_split = round(int(img_h*color_props[1]*0.01))
        img[:first_split, index*acid_width:index*acid_width*2, :] = white
        img[first_split:first_split+second_split, index*acid_width:index*acid_width*2, :] = gray
        img[first_split+second_split:, index*acid_width:index*acid_width*2, :] = black
    return img

"""
Model to train protein sequences of len btw 0-300 and samples size 1k-10k
"""
import sys, random , re, os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.config import CfgNode as CN
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

ProjectRoot = Path('/home/satish27may/ProteinDomainDetection/')
print(f"ProjectRoot: {ProjectRoot}")
sys.path.append(str(ProjectRoot))

from src.data.modelData import ObjectDetection, create_protein_seq_image


# create bucket data
seq_len_bucket = (0, 300)
num_sample_bucket = (1000,10000)
img_h, img_w = 224, seq_len_bucket[1]
config_name = f"seq_len_{seq_len_bucket[0]}-{seq_len_bucket[1]}_and_num_samples_{num_sample_bucket[0]}-{num_sample_bucket[1]}"

os.system(f"trash-put {str(ProjectRoot/f'data/PfamData/{config_name}_images')}")
os.system(f"trash-put {str(ProjectRoot/f'models/{config_name}_model')}")

# create model dir to save checkpoints
all_models_dir = ProjectRoot/'models'
if not all_models_dir.exists():
    all_models_dir.mkdir(exist_ok=True, parents=True)
model_dir = ProjectRoot/f'models/{config_name}_model'
model_dir.mkdir(exist_ok=True, parents=True)

# # log to log file
# log_file = model_dir/'logs.txt'
# sys.stdout = open(str(log_file), 'w')
    
# create all classes data from pfam fasta data
all_classes = ['Lysozyme-PF03245',
'Lysozyme-PF16754',
'Lysozyme-PF11860',
'Lysozyme-PF13702',
'Lysozyme-PF00959',
'Lysozyme-PF00182',
'Lysozyme-PF00704',
'Lysozyme-PF01374',
'Lysozyme-PF05838',
'Lysozyme-PF18013',
'Lysozyme-PF04965',
'Lysozyme-PF01183',
'Lysozyme-PF00722',
'peptidase-PF05193',
'peptidase-PF01551',
'peptidase-PF00675',
'peptidase-PF01435',
'peptidase-PF01433',
'peptidase-PF10502',
'peptidase-PF00246',
'peptidase-PF03572',
'peptidase-PF00814',
'peptidase-PF17900',
'Amidase_2-PF01510',
'Amidase_3-PF01520',
'CHAP-PF05257',
'SH3_4-PF06347',
'SH3_3-PF08239',
'SH3_5-PF08460',
'LysM-PF01476']
data_handler = ObjectDetection(class_names=all_classes)
protein_domain_data = data_handler.create_protein_domain_df()

# create data df for choosen config
bucket_df = data_handler.get_bucketised_data(protein_domain_data, seq_len_bucket, num_sample_bucket)
class_freq_map = dict(bucket_df['Class'].value_counts())
classes = [cls for cls in list(bucket_df['Class'].unique()) if class_freq_map[cls]>50]
bucket_df = bucket_df[bucket_df['Class'].isin(classes)]
# select sub set classes !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
bucket_df = bucket_df[bucket_df['Class'].isin(['PF08239', 'PF06347', 'PF00246', 'PF01510'])]
classes= list(bucket_df['Class'].unique() )
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
bucket_df.to_csv(ProjectRoot/f"data/PfamData/seq_len_{'-'.join([str(x) for x in seq_len_bucket])}_and_num_samples_{'-'.join([str(x) for x in num_sample_bucket])}_data.csv",index=False)


# create image data for choosen config
data_handler.create_bucket_sequence2histogram(bucket_df, seq_len_bucket, num_sample_bucket)

# print config data summary
images_dir = ProjectRoot/f"data/PfamData/{config_name}_images"
model_data =  pd.read_csv(ProjectRoot/f"data/PfamData/{config_name}_data.csv")
print(f"Selected {model_data['Class'].nunique()} classes:\n{model_data['Class'].unique()}")

###############################################Create Model Data##############################################################################


def create_train_valid_test_data(data_df):
    train_dfs,valid_dfs = [],[],
    for class_name in data_df['Class'].unique():
        class_df = data_df[data_df['Class']==class_name].sample(frac=1)
        num_samples = class_df.shape[0]
        num_train_samples = int(round(num_samples*0.7))
        train_dfs.append(class_df.iloc[:num_train_samples,:])
        valid_dfs.append(class_df.iloc[num_train_samples:,:])
    return pd.concat(train_dfs,axis='rows').sample(frac=1), pd.concat(valid_dfs,axis='rows').sample(frac=1)
    
def create_dataset(model_data, img_h, img_w, mode, classes, aug_data):
    model_data = model_data[model_data['Class'].isin([x.split('-')[-1] for x in classes])]
    print(f"Classes selected: {model_data['Class'].unique()}")
    print(f"SuperClasses selected: {model_data['SuperClass'].unique()}")
    model_data = model_data.reset_index(drop=True)
    model_data['dom_pos'] = model_data['dom_pos'].apply(lambda x: [int(y) for y in x.replace('[','').replace(']','').split(',')])
    C2I = {class_name:index for index, class_name in enumerate(model_data['Class'].unique())}
    train_dicts_list = []
    valid_dicts_list = []
    train, valid = create_train_valid_test_data(model_data)
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    if mode=='train':
        for index in tqdm(range(train.shape[0])):
            x1,x2 = train['dom_pos'][index]
            train_dicts_list.append({'file_name':train['img_pth'][index],
                               'height':img_h,
                               'width': img_w,
                               'image_id': index,
                               'annotations':[{'bbox':[x1, 0, x2, img_h],
                                               'bbox_mode':BoxMode.XYXY_ABS,
                                               'category_id':  C2I[train['Class'][index]],
                                              }]
                              })
    if mode=='valid':
        for index in tqdm(range(valid.shape[0])):
            x1,x2 = valid['dom_pos'][index]
            valid_dicts_list.append({'file_name':valid['img_pth'][index],
                               'height':img_h,
                               'width': img_w,
                               'image_id': index,
                               'annotations':[{'bbox':[x1, 0, x2, img_h],
                                               'bbox_mode':BoxMode.XYXY_ABS,
                                               'category_id':  C2I[valid['Class'][index]],
                                              }]
                              })
    if mode=='train':return train_dicts_list
    elif mode=='valid': return valid_dicts_list

# create train and valid data
train_list = create_dataset(model_data, img_h=img_h, img_w=img_w, mode='train', classes=classes, aug_data=False)
valid_list = create_dataset(model_data, img_h=img_h, img_w=img_w, mode='valid', classes=classes, aug_data=False)
def get_train_data():
    return train_list

def get_valid_data():
    return valid_list


# add custom dataset to detectron 2 pipline
DatasetCatalog.register("train", get_train_data)
DatasetCatalog.register("valid", get_valid_data)

train_meta_data = MetadataCatalog.get("train").set(thing_classes = classes)
valid_meta_data = MetadataCatalog.get("valid").set(thing_classes = classes)

##############################################################################################################################################
# train model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("valid",)
#cfg.MODEL.PIXEL_MEAN = data_mean
#cfg.MODEL.PIXEL_STD = data_std
cfg.INPUT.RANDOM_FLIP = "vertical"
cfg.TEST.DETECTIONS_PER_IMAGE = 100

cfg.INPUT.MIN_SIZE_TRAIN = (img_h,)
cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
# Maximum size of the side of the image during training
cfg.INPUT.MAX_SIZE_TRAIN = img_w
# Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
cfg.INPUT.MIN_SIZE_TEST = img_h
# Maximum size of the side of the image during testing
cfg.INPUT.MAX_SIZE_TEST = img_w


cfg.TEST.AUG.FLIP = False
cfg.DATALOADER.NUM_WORKERS = 8
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")  
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 1e-3  
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.RETINANET.NUM_CLASSES = len(classes)


# exp configs
cfg.MODEL.FPN.IN_FEATURES = ["64", "128", "256", "512"]


cfg.OUTPUT_DIR =str(model_dir)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# evaluate model
evaluator = COCOEvaluator("valid", ("bbox",), False, output_dir=cfg.OUTPUT_DIR )
val_loader = build_detection_test_loader(cfg, "valid")