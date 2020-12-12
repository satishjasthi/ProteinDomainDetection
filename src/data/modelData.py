import random, multiprocessing, string
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from Bio import SeqIO
from functools import partial
from imgaug import augmenters as iaa
from tqdm import tqdm

ProjectRoot = Path(__file__).resolve().parent.parent.parent


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
                acid_color_prop_maps[acid] = (first_prop, second_prop, third_prop)

    return acid_color_prop_maps


def sequenc2histogram(seq, img_h, img_w):
    """
    Converts seq to image inthe form of hist
    """
    acid_color_prop_maps = generate_acid_color_proportions()
    acid_width = 1
    # img_h = img_h
    img_w = len(seq)*acid_width
    img = np.full((img_h, img_w, 3),(500,500,500))
    for index in range(img_w):
        if index < len(seq):
            acid = seq[index]
            color_props = acid_color_prop_maps[acid]
            white = (220,220,220)
            black = (0,0,0)
            gray = (90,90,90)
            first_split = round(int(img_h*color_props[0]*0.01))
            second_split = round(int(img_h*color_props[1]*0.01))
            img[:first_split, index*acid_width:index*acid_width*2, :] = white
            img[first_split:first_split+second_split, index*acid_width:index*acid_width*2, :] = gray
            img[first_split+second_split:, index*acid_width:index*acid_width*2, :] = black
        else:
            img[:,index,:] = (255,255,255)
    return img

def generate_sequence2histogram_images(item=None):
    
    sequence,img_name, img_h, img_w = item
    image = sequenc2histogram(sequence, img_h, img_w )
    image = Image.fromarray(image.astype(np.uint8))
    image.save(img_name)

def create_protein_seq_image(item=None, color_map=None):
    """
    Create an image from a sequence
    """
    sequence,img_name, img_h, img_w = item
    image = np.full((img_h, img_w,3), (500,500,500))
    for index in range(len(sequence)):
        image[:, index, :] = color_map[sequence[index]]
    pil_image = Image.fromarray(image.astype(np.uint8))
    assert pil_image.size == (img_w, img_h), f"{pil_image.size}!=({img_w},{img_h})"
    pil_image.save(img_name)

        

class ObjectDetection:
    
    def __init__(self, class_names:list=None, config_name=None):
        self.class_names = class_names
        self.data_dir = Path(__file__).resolve().parent.parent.parent/'data'
        self.config_name = config_name
        self.config_images_dir = ProjectRoot/f'data/PfamData/{self.config_name}_images'
        if not self.config_images_dir.exists():
            self.config_images_dir.mkdir(parents=True, exist_ok=True)

             
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
    def add_gaussian_noise(img_pth,dim, index)->str:
        seq = iaa.Sequential([iaa.SaltAndPepper(0.05)])
        img_arrs = np.zeros((2,dim[0], dim[1], 3))
        # print(f"img_arrsshape: {img_arrs.shape} ")
        # print(f"img_pth:{img_pth}")
        # print(f"img shape:{np.array(Image.open(img_pth)).shape}")
        img_arrs[0,:,:,:] = np.array(Image.open(img_pth))
        images_aug = seq(images=img_arrs)[0]
        aug_img_pth = img_pth.parent/f'{img_pth.stem}_gaussian_noise_{index}.png'
        Image.fromarray(images_aug.astype(np.uint8)).save(aug_img_pth)
        return str(aug_img_pth)
    
    
    @staticmethod
    def add_cutout(img_pth, dim, index)->str:
        seq = iaa.Sequential([iaa.Cutout(nb_iterations=(10, 20), size=0.05, squared=False)])
        img_arrs = np.zeros((2,dim[0], dim[1], 3))
        img_arrs[0,:,:,:] = np.array(Image.open(img_pth))
        images_aug = seq(images=img_arrs)[0]
        aug_img_pth = img_pth.parent/f'{img_pth.stem}_cutout_{index}.png'
        Image.fromarray(images_aug.astype(np.uint8)).save(aug_img_pth)
        return str(aug_img_pth)
                
    def handle_multiple_domains(self, data_df):
        print('Handling sequences with multiple domains')
        # get sequences with multiple domains
        multi_domain_seqs = data_df[data_df.duplicated(subset=['Sequence'])]['Sequence']
        records = []
        for seq in tqdm(multi_domain_seqs):
            seq_df = data_df[data_df['Sequence']==seq]
            dom_pos_list = seq_df['dom_pos']
            dom_len_list = seq_df['dom_len']
            dom_list = seq_df['dom']
            records.append({'id':seq_df['id'].unique()[0],
                            'name': seq_df['name'].unique()[0],
                            'Sequence': seq_df['Sequence'].unique()[0],
                            'Class': '_'.join(seq_df['Class'].unique()),
                            'SeqLen': seq_df['SeqLen'].unique()[0],
                            'SuperClass': '_'.join(seq_df['SuperClass'].unique()),
                            'img_pth': seq_df['img_pth'].unique()[0],
                            'dom_pos': dom_pos_list,
                            'dom_len': dom_len_list,
                            'dom': dom_list,
                            })
        # drop all duplicates
        data_df.drop_duplicates(subset=['Sequence'], keep=False, inplace=True)
        data_df = pd.concat([data_df, pd.DataFrame(data=records)],axis='rows')#.sample(frac=1)
        #data_df.reset_index(drop=True, inplace=True)
        return data_df
        
        

    def create_protein_domain_df(self):
        data = {}
        for clss in self.class_names:
            # print(f'Fetching data from FASTA files for {clss}')
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
                data[record.id]['SeqLen']= len(record.seq._data)
                data[record.id]['SuperClass'] = cls_name
                assert 'PF' in data[record.id]['Class']
                data[record.id]['dom_pos'] = []
                data[record.id]['dom_len'] = []
                data[record.id]['dom'] = []
                data[record.id]['img_pth'] = self.config_images_dir/f"img_{record.id}_{data[record.id]['Class']}_{ data[record.id]['SuperClass']}.png"
                
            for record in SeqIO.parse(domain_data, 'fasta'):
                id_ = record.id.split('/')[0]
                # to ensure the indexing of the domains starts at 0 1 is sub
                # if id_ == 'A0A1B0T6L3_9CAUD':
                #     print('h')
                data[id_]['dom_pos'].append([int(pos)-1 for  pos in record.id.split('/')[-1].split('-')] )
                data[id_]['dom_len'].append(len(record.seq._data))
                data[id_]['dom'].append(record.seq._data)
                
        data_df = pd.DataFrame(data=data.values())
        tot_num_rows = data_df.shape[0]
        # because this class has just one sample
        data_df = data_df[data_df['Class']!='PF13647']
   
        data_df['SeqLen'] = data_df['Sequence'].apply(lambda x: len(x))
        # so that none of the domain pos is missing
        assert data_df['dom_pos'].isna().sum() == 0
        num_total_rows = data_df.shape[0]
        # didnt drop duplicates because they will have multiple domains
        # data_df = data_df.drop_duplicates(subset=['Sequence'])
        # print(f'Dropped {num_total_rows - data_df.shape[0]} number of duplicates based on sequences')
        
        data_df.to_csv(self.data_dir/f'PfamData/model_data.csv',index=False)
        # print('Class distribution: \n')
        # print(data_df['Class'].value_counts())
        return data_df
    
    @staticmethod
    def get_bucketised_data( data_df, seq_len_bucket, num_samples_bucket):
        print(f"Fetching data with sequence len in btw {seq_len_bucket} and class with num samples btw {num_samples_bucket}")       
        # filter data based on seq len
        seq_len_df = data_df[(data_df['SeqLen']>=seq_len_bucket[0]) & (data_df['SeqLen']<seq_len_bucket[1])]
        assert max(seq_len_df['SeqLen']) < seq_len_bucket[1] and min(seq_len_df['SeqLen']) >= seq_len_bucket[0]
        class_freq_map = dict(seq_len_df['Class'].value_counts())
        # filter based on sample size for classes
        required_classes = [class_name for class_name in class_freq_map.keys() if class_freq_map[class_name]>=num_samples_bucket[0] and class_freq_map[class_name]<num_samples_bucket[1]]
        samples_df = seq_len_df[seq_len_df['Class'].isin(required_classes)]
        assert sum([1 for class_name in  required_classes if class_freq_map[class_name]>=num_samples_bucket[0] and class_freq_map[class_name]<num_samples_bucket[1]]) == len(required_classes)
        print(f"Bucket class distribution: {samples_df['Class'].value_counts()}\n  Sequences lens: min={samples_df['SeqLen'].min()}, max = {samples_df['SeqLen'].max()}")
        return samples_df


    def create_bucket_image_data(self, data_df, seq_len_bucket, num_samples_bucket):
        img_h, img_w = 64, seq_len_bucket[1]
        print(f'Genreating  images of dim {img_h}x{img_w} data bucket with sequence len in btw {seq_len_bucket} and class with num samples btw {num_samples_bucket}')
        partial_create_protein_seq_image = partial(create_protein_seq_image,color_map=self.color_map)
        images_folder_path = ProjectRoot/f"data/PfamData/seq_len_{'-'.join([str(x) for x in seq_len_bucket])}_and_num_samples_{'-'.join([str(x) for x in num_samples_bucket])}_images"
        if not images_folder_path.exists():
            images_folder_path.mkdir(parents=True, exist_ok=True)
        data_df['img_pth'] = data_df['img_pth'].apply(lambda x: str(x).replace('images_224', f"seq_len_{'-'.join([str(x) for x in seq_len_bucket])}_and_num_samples_{'-'.join([str(x) for x in num_samples_bucket])}_images"))
        items = [(sequence, img_name, img_h, img_w ) for sequence, img_name in zip(data_df['Sequence'],data_df['img_pth'])]
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as p:
            p.map(partial_create_protein_seq_image,items)
            
    def create_class_image_data(self, data_df, max_seq_len, class_name):
        img_h, img_w = 64, max_seq_len
        print(f'Genreating  images of dim {img_h}x{img_w} for {class_name} data')
        partial_create_protein_seq_image = partial(create_protein_seq_image,color_map=self.color_map)
        images_folder_path = ProjectRoot/f"data/PfamData/{class_name}_images"
        if not images_folder_path.exists():
            images_folder_path.mkdir(parents=True, exist_ok=True)
        data_df['img_pth'] = data_df['img_pth'].apply(lambda x: str(x).replace('images_224', f"{class_name}_images"))
        items = [(sequence, img_name, img_h, img_w ) for sequence, img_name in zip(data_df['Sequence'],data_df['img_pth'])]
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as p:
            p.map(partial_create_protein_seq_image,items)
            
    def create_bucket_sequence2histogram(self, data_df, seq_len_bucket, num_samples_bucket):
        img_h, img_w = 224, seq_len_bucket[1]
        print(f'Genreating  images of dim {img_h}x{img_w} data bucket with sequence len in btw {seq_len_bucket} and class with num samples btw {num_samples_bucket}')
        images_folder_path = ProjectRoot/f"data/PfamData/seq_len_{'-'.join([str(x) for x in seq_len_bucket])}_and_num_samples_{'-'.join([str(x) for x in num_samples_bucket])}_images"
        if not images_folder_path.exists():
            images_folder_path.mkdir(parents=True, exist_ok=True)
        data_df['img_pth'] = data_df['img_pth'].apply(lambda x: str(x).replace('images_224', f"seq_len_{'-'.join([str(x) for x in seq_len_bucket])}_and_num_samples_{'-'.join([str(x) for x in num_samples_bucket])}_images"))
        items = [(sequence, img_name, img_h, img_w ) for sequence, img_name in zip(data_df['Sequence'],data_df['img_pth'])]
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as p:
            p.map(generate_sequence2histogram_images,items)
            
    def create_yolo_sequence2histogram(self, data_df, img_h, img_w):
        print(f'Genreating  images of dim {img_h}x{img_w} data ')
        images_folder_path = ProjectRoot/f"Yolo/yolov5/data/PfamData/images"
        if not images_folder_path.exists():
            images_folder_path.mkdir(parents=True, exist_ok=True)
        data_df['img_pth'] = data_df['img_pth'].apply(lambda x: Path('/home/satish27may/ProteinDomainDetection/Yolo/yolov5/data/PfamData/images')/f"{Path(x).name}")
        items = [(sequence, img_name, img_h, img_w ) for sequence, img_name in zip(data_df['Sequence'],data_df['img_pth'])]
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as p:
            p.map(generate_sequence2histogram_images,items)
            
    
    def augment_data(self, data_df,dim, num_augs):
        class_freq = dict(data_df['Class'].value_counts())
        aug_rows = []
        for class_name in data_df['Class'].unique():
            num_augs_req = abs(num_augs - class_freq[class_name])
            print(f"{class_name} has {class_freq[class_name]} samples, adding {num_augs_req} augs..")
            class_df = data_df[data_df['Class']==class_name]
            rows = [row for _, row in class_df.iterrows()]
            for index in range(round(int((num_augs_req/2)))):
                curr_row = random.choice(rows)
                img_pth = Path(curr_row['img_pth'])
                gaussian_noise = curr_row.copy()
                gaussian_noise['img_pth'] = self.add_gaussian_noise(img_pth,dim, index)
                cutout = curr_row.copy()
                cutout['img_pth'] = self.add_cutout(img_pth,dim, index)
                aug_rows.extend([cutout, gaussian_noise])
                    
        augs_df = pd.DataFrame(data=aug_rows)
        data_df = pd.concat([data_df, augs_df], axis='rows')
        return data_df
                    
            
if __name__ == "__main__":
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

    # SH3_4 and CHAP
    # all_classes= ['SH3_4-PF06347']
    # Lysozyme and SH3_4
    # subset_classes = ['PF01374', 'PF06347']
    data_handler = ObjectDetection(class_names=all_classes)
    protein_domain_data = data_handler.create_protein_domain_df()
    protein_domain_data.to_csv('/home/satish27may/ProteinDomainDetection/trash/protein_data.csv',index=False)

    # # create data df for choosen config
    # bucket_df = data_handler.get_bucketised_data(protein_domain_data, (0,300), (1000, 10000))
    # class_freq_map = dict(bucket_df['Class'].value_counts())
    # classes = [cls for cls in list(bucket_df['Class'].unique()) if class_freq_map[cls]>50]
    # bucket_df = bucket_df[bucket_df['Class'].isin(classes)]
    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # bucket_df = bucket_df[bucket_df['Class'].isin(subset_classes)]
    # classes = list(bucket_df['Class'].unique())
    # data_handler.create_bucket_image_data(bucket_df, (0,300), (1000, 10000))
    
    'SH3_3-PF08239','peptidase-PF01433','Lysozyme-PF01183','Lysozyme-PF05838','Lysozyme-PF01374','Lysozyme-PF11860'
    'Lysozyme-PF00182','CHAP-PF05257','Lysozyme-PF04965','peptidase-PF05193','Lysozyme-PF00959','SH3_4-PF06347',
    'Lysozyme-PF13702','Lysozyme-PF03245','Lysozyme-PF18013'












PF18013      403
