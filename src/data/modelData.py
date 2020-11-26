import random, multiprocessing, string
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from Bio import SeqIO
from functools import partial
from imgaug import augmenters as iaa


def create_protein_seq_image(item=None, color_map=None):
    """
    Create an image from a sequence
    """
    sequence,img_name, img_h, img_w = item
    image = np.full((img_h, img_w,3), (500,500,500))
    for index in range(len(sequence)):
        image[:, index, :] = color_map[sequence[index]]
    pil_image = Image.fromarray(image.astype(np.uint8))
    pil_image.save(img_name)

        

class ObjectDetection:
    
    def __init__(self, class_names:list=None, img_dim=None):
        self.class_names = class_names
        self.img_dim = img_dim
        self.data_dir = Path(__file__).resolve().parent.parent.parent/'data'
        self.images_dir = self.data_dir/f'PfamData/images_{self.img_dim}'
        if not self.images_dir.exists():
            self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir = self.data_dir/f'PfamData/annotations_{self.img_dim}'
        if not  self.annotations_dir.exists():
             self.annotations_dir.mkdir(parents=True, exist_ok=True)
             
        # color map
        self.color_map = {}
        index = 0
        for amino_acid in string.ascii_uppercase:
            self.color_map[amino_acid] = (index+10, index+10, index+10)
            index = index+10
        assert len(set(self.color_map.values())) == len(string.ascii_uppercase)
             
    @staticmethod
    def create_train_valid_test_data(data_df):
        train_dfs,valid_dfs = [],[],
        for class_name in data_df['Class'].unique():
            class_df = data_df[data_df['Class']==class_name].sample(frac=1)
            num_samples = class_df.shape[0]
            num_train_samples = int(round(num_samples*0.7))
            train_dfs.append(class_df.iloc[:num_train_samples,:])
            valid_dfs.append(class_df.iloc[num_train_samples:,:])
        return pd.concat(train_dfs,axis='rows'), pd.concat(valid_dfs,axis='rows')
    
    @staticmethod
    def add_gaussian_noise(img_pth, index)->str:
        seq = iaa.Sequential([iaa.SaltAndPepper(0.05)])
        img_arrs = np.zeros((2,224, 1000, 3))
        img_arrs[0,:,:,:] = np.array(Image.open(img_pth))
        images_aug = seq(images=img_arrs)[0]
        aug_img_pth = img_pth.parent/f'{img_pth.stem}_gaussian_noise_{index}.png'
        Image.fromarray(images_aug.astype(np.uint8)).save(aug_img_pth)
        return str(aug_img_pth)
    
    
    @staticmethod
    def add_cutout(img_pth, index)->str:
        seq = iaa.Sequential([iaa.Cutout(nb_iterations=(10, 20), size=0.05, squared=False)])
        img_arrs = np.zeros((2,224, 1000, 3))
        img_arrs[0,:,:,:] = np.array(Image.open(img_pth))
        images_aug = seq(images=img_arrs)[0]
        aug_img_pth = img_pth.parent/f'{img_pth.stem}_cutout_{index}.png'
        Image.fromarray(images_aug.astype(np.uint8)).save(aug_img_pth)
        return str(aug_img_pth)
                

    
    def augment_data(self, data_df, num_augs):
        if data_df.shape[0] < num_augs:
            aug_rows = []
            rows = [row for _, row in data_df.iterrows()]
            num_augs_required = abs(data_df.shape[0] - num_augs)
            print(f"Generating {num_augs_required} augs for {list(data_df['Class'].unique())[0]}")
            records = [random.choice(rows) for _ in range(num_augs_required)]
            for index, row in enumerate(records):
                img_pth = row['img_pth']
                gaussian_noise = row.copy()
                gaussian_noise['img_pth'] = self.add_gaussian_noise(img_pth, index)
                cutout = row.copy()
                cutout['img_pth'] = self.add_cutout(img_pth, index)
                aug_rows.extend([cutout, gaussian_noise])
            aug_df = pd.DataFrame(data=aug_rows)
            # aug_df = aug_df.drop_duplicates()
            # print(f'Number of augs remaining after removing duplicates: {num_augs_required - aug_df.shape[0]}')
            return pd.concat([data_df,aug_df],axis='rows').sample(frac=1)
        
        else:
            return data_df
    
            
    
            
    @staticmethod
    def create_coco_annotations(df,C2I, img_h, img_w):
        json_data = {   
    "info": {
            "year": "2020",
            "version": "1",
            "description": "Protein Domain annotation data",
            "contributor": "SatishJasthi",
            "url": None,
            "date_created": None
        },
        "licenses": [
            {
                "id": None,
                "url": None,
                "name": None
            }
        ],
        "categories":[],
        "images":[],
        "annotations":[],
        }
        # add categories
        for index, class_name in enumerate(df['Class'].unique()):
            json_data["categories"].append({"id": index+1,"name": class_name,"supercategory": "ProteinDomain"})
            
        # add images and annotations
        for index, row in df.iterrows():

            json_data["images"].append({
                "id": index,
                "license": None,
                "file_name": Path(row['img_pth']).name,
                "height": img_h,
                "width": img_w,
                "date_captured": None,
                "dom_pos":row["dom_pos"]
            })
            json_data['annotations'].append({
                "id": index,
                "image_id": index,
                "category_id": C2I[row['Class']]+1,
                "bbox": [
                    row['dom_pos'][0],
                    0,
                    (row['dom_pos'][1])-(row['dom_pos'][0]),
                    img_h
                ],
                "area": img_h*((row['dom_pos'][1])-(row['dom_pos'][0])),
                "iscrowd":0

            })
        return json_data
    
    def create_coco_data(self, augment_data=False, num_augs=2000, img_h=224, img_w = 1000):
        """
        Method to create both images and coco annotations using protein sequences
        """
        data = {}
        for clss in self.class_names:
            print(f'Fetching data from FASTA files for {clss}')
            cls_name, sub_cls_id = clss.split('-')
            full_sequence_data = self.data_dir/f'PfamData/{cls_name}___full_sequence_data/{cls_name}-{sub_cls_id}___full_sequence_data.fasta'
            domain_data = self.data_dir/f'PfamData/{cls_name}___full_sequence_data/{cls_name}-{sub_cls_id}___domain_data.fasta'     
            
            # extract data from fasta files
            for record in SeqIO.parse(full_sequence_data,'fasta'):
                data[record.id] = {}
                data[record.id]['id'] = record.id
                data[record.id]['name'] = record.name
                data[record.id]['Sequence'] = record.seq._data
                data[record.id]['Class'] = sub_cls_id
                data[record.id]['SuperClass'] = cls_name
                assert 'PF' in data[record.id]['Class']
                data[record.id]['img_pth'] = self.images_dir/f"img_{record.id}_{data[record.id]['Class']}_{ data[record.id]['SuperClass']}.png"
                
            for record in SeqIO.parse(domain_data, 'fasta'):
                id_ = record.id.split('/')[0]
                # to ensure the indexing of the domains starts at 0 1 is sub
                data[id_]['dom_pos'] = [int(pos)-1 for  pos in record.id.split('/')[-1].split('-')]   
                data[id_]['dom'] = record.seq._data
                
        data_df = pd.DataFrame(data=data.values())
        tot_num_rows = data_df.shape[0]
        data_df = data_df[data_df['Class']!='PF13647']
        data_df['SeqLen'] = data_df['Sequence'].apply(lambda x: len(x))
        data_df = data_df[data_df['SeqLen']<1000]
        print(f"number of seqs above 1000: {tot_num_rows - data_df.shape[0]}")
        assert data_df['dom_pos'].isna().sum() == 0
        num_total_rows = data_df.shape[0]
        data_df = data_df.drop_duplicates(subset=['Sequence'])
        print(f'Dropped {num_total_rows - data_df.shape[0]} number of duplicates based on sequences')
        data_df.to_csv(self.data_dir/f'PfamData/model_data.csv',index=False)
        print('Class distribution: \n')
        print(data_df['Class'].value_counts())
        train_df, valid_df = self.create_train_valid_test_data(data_df)
        print('Genreating  images')
        partial_create_protein_seq_image = partial(create_protein_seq_image,color_map=self.color_map)
        items = [(sequence, img_name, img_h, img_w ) for sequence, img_name in zip(data_df['Sequence'],data_df['img_pth'])]
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as p:
            p.map(partial_create_protein_seq_image,items)
        if augment_data:
            print('Augmenting data..........')
            train_df = pd.concat([self.augment_data(train_df[train_df['Class']==clss],num_augs) for clss in train_df['Class'].unique()], axis='rows',sort=False)
            train_df = train_df.reset_index(drop=True)
        
                
        self.C2I = {class_name:index for index, class_name in enumerate(data_df['Class'].unique())}

        # create annotations data
        print('Saving annotations')
        import json
        train_annotations = self.create_coco_annotations(train_df,self.C2I,img_h,img_w)
        valid_annotations = self.create_coco_annotations(valid_df,self.C2I,img_h,img_w)
        with open(self.annotations_dir/'train.json', 'w') as fp:
            json.dump(train_annotations, fp)
        with open(self.annotations_dir/'val.json', 'w') as fp:
            json.dump(valid_annotations, fp)
            
            
            
            
if __name__ == "__main__":
    o = ObjectDetection(class_names=['Lysozyme_PF16754', 'Lysozyme_PF18013', 'Lysozyme_PF01374', 'Lysozyme_PF13702', 'Lysozyme_PF11860', 'Lysozyme_PF03245'], img_dim=224)
    o.create_coco_data(augment_data=True, num_augs=1000, img_h=224, img_w=1000)