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

ProjectRoot = Path(__file__).resolve().parent.parent.parent
print(f"ProjectRoot: {ProjectRoot}")
sys.path.append(str(ProjectRoot))

from src.data.modelData import ObjectDetection, create_protein_seq_image



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
# bucket_df = bucket_df[bucket_df['Class'].isin(['PF08239', 'PF06347', 'PF00246', 'PF01510'])]
# classes= list(bucket_df['Class'].unique() )
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
bucket_df.to_csv(ProjectRoot/f"data/PfamData/seq_len_{'-'.join([str(x) for x in seq_len_bucket])}_and_num_samples_{'-'.join([str(x) for x in num_sample_bucket])}_data.csv",index=False)


# create image data for choosen config
data_handler.create_bucket_image_data(bucket_df, seq_len_bucket, num_sample_bucket)

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
        # num_train_samples = int(round(num_samples*0.7))
        num_train_samples = 500
        num_valid_samples = 200
        train_dfs.append(class_df.iloc[:num_train_samples,:])
        # valid_dfs.append(class_df.iloc[num_train_samples:,:])
        valid_dfs.append(class_df.iloc[num_train_samples:num_train_samples+num_valid_samples,:])
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
    if aug_data:
        print(f"train data before aug: {train.shape[0]}\n{train['Class'].value_counts()}")
        train = data_handler.augment_data(train,(img_h,img_w), num_augs=300)
        print(f"Class dist in train after aug:\n {train['Class'].value_counts()}")
        print(f'train data after aug: {train.shape[0]}')
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
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

MetadataCatalog.get("train").thing_classes = classes
MetadataCatalog.get("valid").thing_classes = classes



##############################################################################################################################################
# train model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
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
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")  
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 1e-3  
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
#cfg.MODEL.RETINANET.IOU_THRESHOLDS = [0.4, 0.5]
cfg.SOLVER.MAX_ITER = 15000
cfg.MODEL.BACKBONE.FREEZE_AT=1
cfg.MODEL.RETINANET.NUM_CLASSES = len(classes)
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(data_handler.class_names)

cfg.OUTPUT_DIR =str(model_dir)
cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [True, True, True, True]
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# evaluate model
evaluator = COCOEvaluator("valid", ("bbox",), False, output_dir=cfg.OUTPUT_DIR )
val_loader = build_detection_test_loader(cfg, "valid")
print(trainer.test(cfg, trainer.model, evaluator))



# |  category  | #instances   |  category  | #instances   |  category  | #instances   |
# |:----------:|:-------------|:----------:|:-------------|:----------:|:-------------|
# |  PF08239   | 3219         |  PF05257   | 1147         |  PF00959   | 1595         |
# |  PF00182   | 1312         |  PF00704   | 1453         |  PF05838   | 766          |
# |  PF01510   | 4782         |  PF04965   | 2210         |  PF01183   | 2142         |
# |  PF01551   | 6385         |  PF00722   | 4388         |  PF01435   | 5515         |
# |  PF00246   | 948          |  PF00814   | 4742         |  PF06347   | 972          |
# |  PF01520   | 2922         |            |              |            |              |
# |   total    | 44498        |            |              |            |              |

# | category   | AP    | category   | AP     | category   | AP     |
# |:-----------|:------|:-----------|:-------|:-----------|:-------|
# | PF08239    | 5.648 | PF05257    | 1.778  | PF00959    | 11.808 |
# | PF00182    | 3.012 | PF00704    | 7.506  | PF05838    | 3.817  |
# | PF01510    | 7.549 | PF04965    | 46.490 | PF01183    | 6.736  |
# | PF01551    | 2.416 | PF00722    | 7.048  | PF01435    | 16.833 |
# | PF00246    | 4.915 | PF00814    | 14.368 | PF06347    | 3.958  |
# | PF01520    | 5.179 |            |        |            |        |


# | category   | AP     | category   | AP     | category   | AP     |
# |:-----------|:-------|:-----------|:-------|:-----------|:-------|
# | PF08239    | 19.131 | PF01510    | 32.684 | PF00246    | 19.971 |
# | PF06347    | 14.535 |            |        |            |        |