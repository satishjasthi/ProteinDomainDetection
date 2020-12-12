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
        records = []
        domain_data_records=[]
        seq_data_records = []
        for class_name in self.classes:
            cls_img_dir = self.images_dir/f"{class_name}"
            cls_img_dir.mkdir(exist_ok=True, parents=True)
            super_class, class_id = class_name.split('-')
            full_seq_data = self.data_dir/f'PfamData/{super_class}___full_sequence_data/{class_name}___full_sequence_data.fasta'
            dom_data = self.data_dir/f'PfamData/{super_class}___full_sequence_data/{class_name}___domain_data.fasta'
            # parse sequences of all classes
            for record in SeqIO.parse(full_seq_data, 'fasta'):
                seq_data_records.append({'Sequence': record.seq._data,
                                        'name': record.name,
                                        'id': record.id,
                                        'Class':class_id,
                                        'SeqLen':len(record.seq._data),
                                        'SuperClass':super_class}
                                      )
            
            # parse domains of all classes
            for record in SeqIO.parse(dom_data, 'fasta'):
                domain_data_records.append({'id':record.id.split('/')[0],
                                            'dom':record.seq._data,
                                            'dom_pos':tuple([int(pos)-1 for  pos in record.id.split('/')[-1].split('-')]),
                                            'dom_len':len(record.seq._data)
                                            })
        seq_data_df = pd.DataFrame(data=seq_data_records)
        domain_data_df = pd.DataFrame(data=domain_data_records)
        all_data = pd.merge(seq_data_df, domain_data_df,how='inner',on='id')
        all_data.drop_duplicates(inplace=True)
        
        for sequence in all_data['Sequence'].unique():
            sequence_df = all_data[all_data['Sequence']==sequence]
            
            records.append({'Sequence':sequence,
                            'Class':'||'.join(sequence_df['Class']),
                            'SuperClass':'||'.join(sequence_df['SuperClass']),
                            'name': '||'.join(sequence_df['name']),
                            'SeqLen':sequence_df['SeqLen'].tolist()[0],
                            'dom':sequence_df['dom'].tolist(),
                            'dom_pos':sequence_df['dom_pos'].tolist(),
                            'dom_len':sequence_df['dom_len'].tolist(),
                            'img_pth':self.images_dir/f"{'_'.join(self.classes)}_{'_'.join(sequence_df['name'])}.png",
                            })
        return pd.DataFrame(data=records)
    
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
        self.cfg.SOLVER.IMS_PER_BATCH = 8
        self.cfg.SOLVER.BASE_LR = 9e-4  
        self.cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
        # self.cfg.SOLVER.MAX_ITER = 3000
        self.cfg.MODEL.RETINANET.NUM_CLASSES = len(classes)

        # exp
        self.cfg.MODEL.RESNETS.NORM = "BN"
        # self.cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA =0.5 #didn't work
        self.cfg.MODEL.RETINANET.IOU_THRESHOLDS = [0.4, 0.99]
        
        self.cfg.OUTPUT_DIR =str(self.model_dir)
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        
    def create_train_valid_data(self, data):
        train_dfs,valid_dfs = [],[],
        for class_name in self.classes:
            class_id = class_name.split('-')[-1]
            class_df = data[data['Class'].str.contains(class_id)].sample(frac=1)
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
        print(f"Classes selected: {self.classes}")
        # print(f"SuperClasses selected: {data['SuperClass'].unique()}")
        data = data.reset_index(drop=True)
        self.C2I = {class_name:index for index, class_name in enumerate(self.classes)}
        dicts_list = []
        data = data.reset_index(drop=True)
        for index in tqdm.tqdm(range(data.shape[0])):
            class_list = data['Class']['index'].split('||')
            pil_img = Image.open(data['img_pth'][index])
            img_w, img_h = pil_img.size
            dom_pos_list = data['dom_pos'][index]
            if len(dom_pos_list)>1:
                annts = []
                for dom_index in range(len(dom_pos_list)):
                    x1,x2 = dom_pos_list[dom_index]
                    annts.append({'bbox':[x1, 0, x2, img_h],
                                            'bbox_mode':BoxMode.XYXY_ABS,
                                            'category_id':  self.C2I[class_list[dom_index]],
                                            })
            elif len(dom_pos_list)==1:
                x1,x2 = dom_pos_list[0]
                annts = [{'bbox':[x1, 0, x2, img_h],
                                            'bbox_mode':BoxMode.XYXY_ABS,
                                            'category_id':  self.C2I[data['Class'][index]],
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
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.99   # set a custom testing threshold
        predictor = DefaultPredictor(self.cfg)
        try:
            # calculate recall 
            tp, fn = 0, 0
            for d in DatasetCatalog.get("valid"):
                im = cv2.imread(str(d["file_name"]))
                outputs = predictor(im)
                true_class_id = d['annotations'][0]['category_id'] 
                try:    
                    pred_class_id = outputs['instances'].get_fields()['pred_classes'][0].cpu().numpy().item()
                except:
                    fn+=1
                    continue
                if true_class_id==pred_class_id:
                    tp+=1
                else:
                    fn +=1
            print(f"Recall for model: {tp/(tp+fn)}")
        except Exception as e:
            print(e)
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
    classes=['CHAP-PF05257', 'SH3_3-PF08239', 'SH3_4-PF06347']# 'SH3_4-PF06347','SH3_5-PF08460']
            # 'SH3_3-PF08239',
            # 'SH3_5-PF08460',
            # 'LysM-PF01476']
    img_h, img_w = 64, 300 # padding is also controlled by this img_w
    seq_len =300   
    seq_buckets = (0, img_w) 

    data_block = Data(classes)
    protein_data = data_block.create_protein_domain_data()
    protein_data = protein_data[protein_data['SeqLen']<seq_len]
    # data_block.create_protein_seq_images(protein_data, img_h, img_w)
    # # data_block.create_protein_seq_len_images(protein_data, img_h, img_w)

    # model_block = Detectron(classes=classes, model_dir=ProjectRoot/f"models/{'_'.join(classes)}_{seq_buckets[0]}_{seq_buckets[1]}")
    # model_block.cfg.SOLVER.MAX_ITER = 3000
    # train_data, valid_data = model_block.create_train_valid_data(protein_data)
    # model_block.register_custom_data(train_data, 'train', img_h, img_w)
    # model_block.register_custom_data(valid_data, 'valid', img_h, img_w)
    # trainer = model_block.train()
    # model_block.evaluate(trainer)
    # predictor = model_block.inference()
    # #class_index = all_classes.index(classes[0])
    # #del all_classes[class_index]
    # model_block.test_for_open_set(protein_data, all_classes, img_h, img_w, predictor)
    
