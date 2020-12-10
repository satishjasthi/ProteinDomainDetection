from functools import partial
import multiprocessing
import os
import random
import string
import sys
from pathlib import Path

from Bio import SeqIO
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from google.colab.patches import cv2_imshow
import tqdm

# ProjectRoot = Path(__file__).resolve().parent.parent.parent
ProjectRoot = Path('/home/satish27may/ProteinDomainDetection')
print(f"ProjectRoot: {str(ProjectRoot)}")

sys.path.append(str(ProjectRoot))

def protein_seq2image(item=None, color_map=None):
    """
    Create an image from a sequence
    """
    sequence,img_name, img_h, img_w = item
    assert len(sequence)<=img_w, f"!! Len of sequence({len(sequence)}) should be less than img_w({img_w}), "
    image = np.full((img_h, img_w,3), (500,500,500))
    for index in range(len(sequence)):
        image[:, index, :] = color_map[sequence[index]]
    pil_image = Image.fromarray(image.astype(np.uint8))
    assert pil_image.size == (img_w, img_h), f"{pil_image.size}!=({img_w},{img_h})"
    pil_image.save(img_name)
    
def protein_seq2image_seqlen(item=None, color_map=None):
    """
    Create an image from a sequence
    """
    sequence,img_name, img_h, img_w = item
    img_w = len(sequence)
    assert len(sequence)<=img_w, f"!! Len of sequence({len(sequence)}) should be less than img_w({img_w}), "
    image = np.full((img_h, img_w,3), (500,500,500))
    for index in range(len(sequence)):
        image[:, index, :] = color_map[sequence[index]]
    pil_image = Image.fromarray(image.astype(np.uint8))
    assert pil_image.size == (img_w, img_h), f"{pil_image.size}!=({img_w},{img_h})"
    pil_image.save(img_name)



class Data:
    
    def __init__(self, classes) -> None:
        super().__init__()
        self.classes = classes
        self.data_dir = ProjectRoot/'data'
        self.images_dir = ProjectRoot/'data/PfamData/protein_seq_images'
        if not self.images_dir.exists():
            self.images_dir.mkdir(parents=True, exist_ok=True)
            
        self.color_map = {}
        index = 0
        for amino_acid in string.ascii_uppercase:
            self.color_map[amino_acid] = (index+10, index+10, index+10)
            index = index+10
        
    def create_protein_domain_data(self):
        data = {}
        for class_name in self.classes:
            cls_img_dir = self.images_dir/f"{class_name}"
            cls_img_dir.mkdir(exist_ok=True, parents=True)
            super_class, class_id = class_name.split('-')
            full_seq_data = self.data_dir/f'PfamData/{super_class}___full_sequence_data/{class_name}___full_sequence_data.fasta'
            dom_seq_data = self.data_dir/f'PfamData/{super_class}___full_sequence_data/{class_name}___domain_data.fasta'
            for record in SeqIO.parse(full_seq_data, 'fasta'):
                data[record.id] = {}
                data[record.id]['id'] = record.id
                data[record.id]['name'] = record.name
                data[record.id]['Sequence'] = record.seq._data
                data[record.id]['Class'] = class_id
                data[record.id]['SeqLen']= len(record.seq._data)
                data[record.id]['SuperClass'] = super_class
                assert 'PF' in data[record.id]['Class']
                data[record.id]['dom_pos'] = []
                data[record.id]['dom_len'] = []
                data[record.id]['dom'] = []
                data[record.id]['img_pth'] = self.images_dir/f"{cls_img_dir}/img_{record.id}_{class_id}_{super_class}.png"
                
            for record in SeqIO.parse(dom_seq_data, 'fasta'):
                id_ = record.id.split('/')[0]
                # to ensure the indexing of the domains starts at 0 1 is sub
                data[id_]['dom_pos'].append([int(pos)-1 for  pos in record.id.split('/')[-1].split('-')] )
                data[id_]['dom_len'].append(len(record.seq._data))
                data[id_]['dom'].append(record.seq._data)
        
        data_df = pd.DataFrame(data=data.values())
        print("Class distribution after creating protein domain data from fasta files: \n")
        print(data_df['Class'].value_counts())
        assert data_df['dom_pos'].isna().sum() == 0
        return data_df
    
    def create_protein_seq_images(self, data_df, img_h, img_w):
        print(f"Generating images of dim {img_h}x{img_w} for classes: {list(data_df['Class'].unique())}")
        for class_name in self.classes:
            if not (self.images_dir/f'{class_name}').exists():
                (self.images_dir/f'{class_name}').mkdir(parents=True, exist_ok=True)
        
        partial_protein_seq2image = partial(protein_seq2image,color_map=self.color_map)
        items = [(sequence, img_name, img_h, img_w ) for sequence, img_name in zip(data_df['Sequence'],data_df['img_pth'])]
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
            p.map(partial_protein_seq2image, items)
            
    def create_protein_seq_len_images(self, data_df, img_h, img_w):
        print(f"Generating images of dim {img_h}x{img_w} for classes: {list(data_df['Class'].unique())}")
        for class_name in self.classes:
            if not (self.images_dir/f'{class_name}').exists():
                (self.images_dir/f'{class_name}').mkdir(parents=True, exist_ok=True)
        
        partial_protein_seq2image_seqlen = partial(protein_seq2image_seqlen,color_map=self.color_map)
        items = [(sequence, img_name, img_h, img_w ) for sequence, img_name in zip(data_df['Sequence'],data_df['img_pth'])]
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
            p.map(partial_protein_seq2image_seqlen, items)
            
class Detectron:
    
    def __init__(self, classes=None, model_dir=None) -> None:
        super().__init__()
        self.classes = classes
        self.model_dir = model_dir
        os.system(f"! trash-put {str(self.model_dir)}")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
        self.cfg.DATASETS.TRAIN = ("train",)
        self.cfg.DATASETS.TEST = ("valid",)
        self.cfg.INPUT.RANDOM_FLIP = "vertical"
        self.cfg.TEST.DETECTIONS_PER_IMAGE = 100

        self.cfg.INPUT.MIN_SIZE_TRAIN = 800
        self.cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
        self.cfg.INPUT.MAX_SIZE_TRAIN = 1330
        self.cfg.INPUT.MIN_SIZE_TEST = 800
        self.cfg.INPUT.MAX_SIZE_TEST = 1330

        self.cfg.TEST.AUG.FLIP = False
        self.cfg.DATALOADER.NUM_WORKERS = 8
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")  
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR = 9e-4  
        self.cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
        # self.cfg.SOLVER.MAX_ITER = 3000
        self.cfg.MODEL.RETINANET.NUM_CLASSES = len(classes)

        # exp
        self.cfg.MODEL.RESNETS.NORM = "BN"
        
        self.cfg.OUTPUT_DIR =str(self.model_dir)
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        
    def create_train_valid_data(self, data):
        train_dfs,valid_dfs = [],[],
        for class_name in data['Class'].unique():
            class_df = data[data['Class']==class_name].sample(frac=1)
            num_samples = class_df.shape[0]
            num_train_samples = int(round(num_samples*0.7))
            train_dfs.append(class_df.iloc[:num_train_samples,:])
            valid_dfs.append(class_df.iloc[num_train_samples:,:])
        train_data = pd.concat(train_dfs,axis='rows').sample(frac=1)
        valid_data = pd.concat(valid_dfs,axis='rows').sample(frac=1)
        print(f'Train data dist:\n')
        print(train_data['Class'].value_counts())
        print(f'Valid data dist:\n')
        print(valid_data['Class'].value_counts())
        assert len(set(list(train_data['id'])).intersection(set(list(valid_data['id']))))==0, f"There is a data leak between train and valid"
        return train_data, valid_data
        
    def register_custom_data(self, data, mode, img_h, img_w):
        print(f"Registring {mode} data.......")
        print(f"Classes selected: {data['Class'].unique()}")
        print(f"SuperClasses selected: {data['SuperClass'].unique()}")
        data = data.reset_index(drop=True)
        C2I = {class_name:index for index, class_name in enumerate(data['Class'].unique())}
        dicts_list = []
        data = data.reset_index(drop=True)
        for index in tqdm.tqdm(range(data.shape[0])):
            pil_img = Image.open(data['img_pth'][index])
            img_w, img_h = pil_img.size
            dom_pos_list = data['dom_pos'][index]
            if len(dom_pos_list)>1:
                annts = []
                for dom_index in range(len(dom_pos_list)):
                    x1,x2 = dom_pos_list[dom_index]
                    annts.append({'bbox':[x1, 0, x2, img_h],
                                            'bbox_mode':BoxMode.XYXY_ABS,
                                            'category_id':  C2I[data['Class'][index]],
                                            })
            elif len(dom_pos_list)==1:
                x1,x2 = dom_pos_list[0]
                annts = [{'bbox':[x1, 0, x2, img_h],
                                            'bbox_mode':BoxMode.XYXY_ABS,
                                            'category_id':  C2I[data['Class'][index]],
                                            }]
                
                    
            dicts_list.append({'file_name':data['img_pth'][index],
                            'height':img_h,
                            'width': img_w,
                            'image_id': index,
                            'annotations':annts,
                            })
        def get_data():
            return dicts_list
        DatasetCatalog.register(mode, get_data)
        MetadataCatalog.get(mode).set(thing_classes = self.classes)
        
    def train(self):
        trainer = DefaultTrainer(self.cfg) 
        trainer.resume_or_load(resume=False)
        trainer.train() 
        return trainer

    def evaluate(self, trainer):
        # evaluate model
        evaluator = COCOEvaluator("valid", ("bbox",), False, output_dir=self.cfg.OUTPUT_DIR )
        val_loader = build_detection_test_loader(self.cfg, "valid")
        print(trainer.test(self.cfg, trainer.model, evaluator))
        
        
    def inference(self):
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.9   # set a custom testing threshold
        predictor = DefaultPredictor(self.cfg)
        dataset_dicts = DatasetCatalog.get("valid")
        # for d in random.sample(dataset_dicts, 3):    
        #     im = cv2.imread(d["file_name"])
        #     outputs = predictor(im)  
        #     v = Visualizer(im[:, :, ::-1],
        #                 metadata=MetadataCatalog.get("train").set(thing_classes = self.classes), 
        #                 scale=1.5, 
        #                 instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        #     )
        #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #     cv2_imshow(out.get_image()[:, :, ::-1])
        return predictor
            
    def test_for_open_set( self, class_df, classes, img_h, img_w, predictor):                                              
        data_handler = Data(classes)                                                                                       
        protein_domain_data = data_handler.create_protein_domain_data()                                                    
        num_rows = protein_domain_data.shape[0]                                                                            
        protein_domain_data = protein_domain_data[~protein_domain_data['Sequence'].isin(class_df['Sequence'])]             
        print(f"Dropping {abs(num_rows - protein_domain_data.shape[0])} sequences which are common with {self.classes}'s sequences")
        protein_domain_data = protein_domain_data[protein_domain_data['SeqLen']<img_w]                        
        data_handler.create_protein_seq_images(protein_domain_data, img_h, img_w) 
        # data_handler.create_protein_seq_len_images(protein_domain_data, img_h, img_w)                           
        all_imgs = protein_domain_data['img_pth'].tolist()                                                    
        print(f"Using {len(all_imgs)} images from classes {classes} for open set recognition test...........")
        count_0_99 = 0                                                                             
        count_0_9 = 0                                                                              
        count_0_8 = 0                                                                                                                                
        count_0_7 = 0                                                                                                                                
        for img_pth in tqdm.tqdm(all_imgs):                                                                                                          
            im = cv2.imread(str(img_pth))                                                                                                            
            outputs = predictor(im)                                                                                                                  
            # v = Visualizer(im[:, :, ::-1],                                                                                                         
            #             metadata=MetadataCatalog.get("valid").set(thing_classes = self.classes),                                                   
            #             scale=0.5,                                                                                                                 
            #             instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models                                                                                                                                     
            # )                                                                                                                              
            # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))                                                              
            # cv2_imshow(out.get_image()[:, :, ::-1])                                                                                        
            try:                                                                                                                             
                instance = outputs['instances']                                                                                              
                if max(instance.get_fields()['scores']).cpu().numpy() >= 0.99:                                                               
                    count_0_99+=1                                                                                                            
                elif max(instance.get_fields()['scores']).cpu().numpy() >= 0.9 and max(instance.get_fields()['scores']).cpu().numpy() < 0.99:
                    count_0_9+=1                                                                                                            
                elif max(instance.get_fields()['scores']).cpu().numpy() >= 0.8 and max(instance.get_fields()['scores']).cpu().numpy() < 0.9:
                    count_0_8+=1                                                                                                            
                elif max(instance.get_fields()['scores']).cpu().numpy() >= 0.7 and max(instance.get_fields()['scores'].cpu().numpy()) < 0.8:
                    count_0_7+=1                                                                                        
            except Exception as e:                                                                                      
                #print(max(instance.get_fields()['scores']).cpu().numpy())                                              
                #print(e)                                                                                               
                pass                                                                                                    
                                                                                                                        
        print(f"Number of predictions with score >0.99 {count_0_99} out of {len(all_imgs)}: {count_0_99/len(all_imgs)}")
        print(f"Number of predictions with score >0.9 {count_0_9} out of {len(all_imgs)}: {count_0_9/len(all_imgs)}")
        print(f"Number of predictions with score >0.8 {count_0_8} out of {len(all_imgs)}: {count_0_8/len(all_imgs)}")
        print(f"Number of predictions with score >0.7 {count_0_7} out of {len(all_imgs)}: {count_0_7/len(all_imgs)}")
      
      
if __name__ == "__main__":
    all_classes = ['Amidase_2-PF01510',
    'Amidase_3-PF01520',]
    classes=['CHAP-PF05257']#SH3_4-PF06347']#]
            # 'SH3_3-PF08239',
            # 'SH3_5-PF08460',
            # 'LysM-PF01476']
    img_h, img_w = 64, 300 # padding is also controlled by this img_w
    seq_len =300    

    data_block = Data(classes)
    protein_data = data_block.create_protein_domain_data()
    protein_data = protein_data[protein_data['SeqLen']<seq_len]
    data_block.create_protein_seq_images(protein_data, img_h, img_w)
    # data_block.create_protein_seq_len_images(protein_data, img_h, img_w)

    model_block = Detectron(classes=classes, model_dir=ProjectRoot/f"models/{'_'.join(classes)}")
    model_block.cfg.SOLVER.MAX_ITER = 3000
    train_data, valid_data = model_block.create_train_valid_data(protein_data)
    model_block.register_custom_data(train_data, 'train', img_h, img_w)
    model_block.register_custom_data(valid_data, 'valid', img_h, img_w)
    trainer = model_block.train()
    model_block.evaluate(trainer)
    predictor = model_block.inference()
    #class_index = all_classes.index(classes[0])
    #del all_classes[class_index]
    model_block.test_for_open_set(protein_data, all_classes, img_h, img_w, predictor)
    
    
#CHAP trained for 15k with all sequences with img_w = 1000 with cosine lr and batch_size =2 and all images are paaded to 1k 
# The logic behind padding is that, since CHAP has big domains even with padding model can identify them 
# |   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
# |:------:|:------:|:------:|:-----:|:------:|:------:|
# | 83.792 | 94.950 | 89.665 |  nan  | 83.825 | 30.000 |
# open set
# Number of predictions with score >0.99 340 out of 22742: 0.014950312197695893
# Number of predictions with score >0.9 1372 out of 22742: 0.06032890686834931
# Number of predictions with score >0.8 0 out of 22742: 0.0
# Number of predictions with score >0.7 0 out of 22742: 0.0

# CHAP trained for 3k with all sequences with img_w = 1000 with cosine lr and batch_size =16 and all images are paaded to 1k 
# The logic behind padding is that, since CHAP has big domains even with padding model can identify them    
# |   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl  |
# |:------:|:------:|:------:|:-----:|:------:|:-----:|
# | 67.965 | 88.976 | 76.249 |  nan  | 67.966 |  nan  |

# CHAP trained for 3k with all sequences with img_w = 1000 with cosine lr and batch_size =2 and all images are paaded to 1k 
# The logic behind padding is that, since CHAP has big domains even with padding model can identify them
# |   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl  |
# |:------:|:------:|:------:|:-----:|:------:|:-----:|
# | 75.542 | 93.449 | 83.593 |  nan  | 75.543 |  nan  |
# open set with al images padded to 1k
# Number of predictions with score >0.99 0 out of 22742: 0.0
# Number of predictions with score >0.9 99 out of 22742: 0.004353179139917333
# Number of predictions with score >0.8 0 out of 22742: 0.0
# Number of predictions with score >0.7 0 out of 22742: 0.0

# CHAP trained for 3k with all sequences with img_w = seq_len with cosine lr
# |   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl  |
# |:------:|:------:|:------:|:-----:|:------:|:-----:|
# | 36.178 | 48.738 | 40.816 |  nan  | 36.188 |  nan  |

# CHAP trained for 3k with all sequences with img_w = seq_len with linear lr
# |   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl  |
# |:------:|:------:|:------:|:-----:|:------:|:-----:|
# | 49.211 | 66.144 | 57.092 |  nan  | 49.223 |  nan  |


# SH3_4 trained for 3k with all sequences with img_w = seq_len
# |   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl  |
# |:------:|:------:|:------:|:-----:|:------:|:-----:|
# | 82.871 | 91.927 | 88.395 |  nan  | 82.875 |  nan  |
# open set with img_w = seq_len and all len sequences
# Number of predictions with score >0.99 0 out of 22830: 0.0
# Number of predictions with score >0.9 88 out of 22830: 0.0038545773105562856
# Number of predictions with score >0.8 0 out of 22830: 0.0
# Number of predictions with score >0.7 0 out of 22830: 0.0



# SH3_4 trained for 3k with < 300 samples with img_w = seq_len
# |   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl  |
# |:------:|:------:|:------:|:-----:|:------:|:-----:|
# | 89.031 | 96.910 | 94.774 |  nan  | 89.034 |  nan  |
# open set with img_w = seq_len and all len sequences
# Number of predictions with score >0.99 5 out of 11478: 0.0004356159609688099
# Number of predictions with score >0.9 268 out of 11478: 0.02334901550792821
# Number of predictions with score >0.8 0 out of 11478: 0.0
# Number of predictions with score >0.7 0 out of 11478: 0.0

# SH3_4 trained for 15k with <300 samples with img_w=300
# |   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl  |
# |:------:|:------:|:------:|:-----:|:------:|:-----:|
# | 91.640 | 96.318 | 94.550 |  nan  | 91.640 |  nan  |
# Number of predictions with score >0.99 363 out of 11478: 0.031625718766335596
# Number of predictions with score >0.9 1713 out of 11478: 0.14924202822791427
# Number of predictions with score >0.8 0 out of 11478: 0.0
# Number of predictions with score >0.7 0 out of 11478: 0.0



# CHAP and SH3_4 trained with <300 sample len for 3k iterations with bs=1 and each image has width = seq_len
# | category      | AP    | category     | AP    |
# |:--------------|:------|:-------------|:------|
# | SH3_4-PF06347 | 9.299 | CHAP-PF05257 | 7.902 |


# CHAP and SH3_4 trained 3k with <300 sample len for 3k iterations with bs=16
# | category      | AP     | category     | AP     |
# |:--------------|:-------|:-------------|:-------|
# | SH3_4-PF06347 | 67.704 | CHAP-PF05257 | 80.064 |
# Number of predictions with score >0.99 166 out of 11478: 0.014462449904164489
# Number of predictions with score >0.9 1601 out of 11478: 0.13948423070221294
# Number of predictions with score >0.8 0 out of 11478: 0.0
# Number of predictions with score >0.7 0 out of 11478: 0.0

# CHAP and SH3_4 trained 3k with <300 sample len for 3k iterations a
# | category      | AP     | category     | AP     |
# |:--------------|:-------|:-------------|:-------|
# | SH3_4-PF06347 | 88.361 | CHAP-PF05257 | 77.538 |
# Number of predictions with score >0.99 0 out of 11478: 0.0
# Number of predictions with score >0.9 42 out of 11478: 0.0036591740721380033
# Number of predictions with score >0.8 0 out of 11478: 0.0
# Number of predictions with score >0.7 0 out of 11478: 0.0

# for SH3_4 single model using all sequences with padding of 1k map=87, 2-3% open set
# for SH3_4 single model using all sequences with padding of 300 map=95, 2-3% open set