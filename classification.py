import string, multiprocessing
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from Bio import SeqIO
from functools import partial
from pandas.core.tools import numeric
from tqdm import tqdm


def create_protein_seq_image(item=None, color_map=None):
    """
    Create an image from a sequence
    """
    sequence,img_name, img_h, img_w = item
    image = Image.new('RGB', (img_w,img_h))
    pixels = image.load()
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            if x < len(sequence):
                pixels[x,y] =  color_map[sequence[x]]
            else:
                pixels[x,y] = (500,500,500)
    try:
        image.save(img_name)
    except:
        print(img_name)
        
class ClassificationData:
    
    def __init__(self, class_names:list=None, img_dim=None):
        self.class_names = class_names
        self.img_dim = img_dim
        self.data_dir = Path(__file__).cwd()/'data'
        self.images_dir = self.data_dir/f'Classification/images_{self.img_dim}'
        if not self.images_dir.exists():
            self.images_dir.mkdir(parents=True, exist_ok=True)
             
        # color map
        self.color_map = {}
        index = 0
        for amino_acid in string.ascii_uppercase:
            self.color_map[amino_acid] = (index+10, index+10, index+10)
            index = index+10
            
    
    
    def create_img_data(self, img_h, img_w):
        train_dir =  self.images_dir/'train'
        val_dir =  self.images_dir/'val'
        if not train_dir.exists():
            train_dir.mkdir(parents=True, exist_ok=True)
        if not val_dir.exists():
            val_dir.mkdir(parents=True, exist_ok=True)
            
        for clss in tqdm(self.class_names):
            cls_name, sub_cls_id = clss.split('_')
            if 'SH3' in clss:
                cls_name = cls_name[:-1] + '_' + cls_name[-1]
            full_sequence_data = self.data_dir/f'PfamData/{cls_name}_full_sequence_data/{cls_name}_{sub_cls_id}_full_sequence_data.fasta'
            sequences = []
            for record in SeqIO.parse(full_sequence_data, 'fasta'):
                sequences.append(record.seq._data)
            num_train_records = int(round(len(sequences)*0.7))
            train_seqs = sequences[:num_train_records]
            val_seqs = sequences[num_train_records:]
            train_class_dir = train_dir/f'{clss}'
            if not train_class_dir.exists():
                train_class_dir.mkdir(parents=True, exist_ok=True)
            val_class_dir = val_dir/f'{clss}'
            if not val_class_dir.exists():
                val_class_dir.mkdir(parents=True, exist_ok=True)
            partial_create_protein_seq_image = partial(create_protein_seq_image, color_map=self.color_map)
            train_items = [(sequence, train_class_dir/f'{clss}_{index}.png', img_h, img_w) for index, sequence in enumerate(train_seqs)]
            with multiprocessing.Pool(processes=8) as p:
                p.map(partial_create_protein_seq_image, train_items)
            
            val_items = [(sequence, val_class_dir/f'{clss}_{index}.png', img_h, img_w) for index, sequence in enumerate(val_seqs)]
            with multiprocessing.Pool(processes=8) as p:
                p.map(partial_create_protein_seq_image, val_items)



        

if __name__ == "__main__":
    o = ClassificationData(class_names=['SH35_PF08460', 'Lysozyme_PF16754', 'Lysozyme_PF18013', 'Lysozyme_PF01374', 'Lysozyme_PF13702', 'Lysozyme_PF11860', 'Lysozyme_PF03245'], img_dim=224)
    o.create_img_data(224,1000)