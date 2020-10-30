"""
Script to train ResNet18 pretrained model on Amidase2, Amidase3 and CHAP full sequence data(Image form)
from Pfam 
"""
import sys
from pathlib import Path
import pandas as pd
from Bio import SeqIO
from fastai.vision import *
from fastai.metrics import error_rate, accuracy
ProjectRoot = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ProjectRoot))
from data.utils import create_image_dataset

class ProteinClassification:
  """
  Class to train classification model on image form of protein 
  domains in protein full sequences to understand whether pretrained models
  can learn to classify the domains in image forms so that they can
  be used in object detection as classification blocks
  """
  def __init__(self, resent_model:str='resnet18'):
    self.data_dir = ProjectRoot/'data'
    self.check_dir_exists(self.data_dir)
    self.clsfn_data = self.data_dir/'classification_data'
    self.check_dir_exists(self.clsfn_data)
    self.train_data_dir = self.clsfn_data/'train'
    self.check_dir_exists(self.train_data_dir)
    self.valid_data_dir = self.clsfn_data/'valid'
    self.check_dir_exists(self.valid_data_dir)
    self.protein_data = self.data_dir/'protein_data'
    self.create_classification_data()

  def create_classification_data(self)->pd.DataFrame:
    """
    Method to create train and valid image data for 
    classification
    """
    # collect full length protein sequence fasta files
    fasta_files = list(self.protein_data.glob('*.fasta'))
    assert len(fasta_files) > 2, f"num fastafiles = {len(fasta_files)}"
    rows = []

    # create dataframe for all class sequences
    for ffile in fasta_files:
      super_class_name, class_name = ffile.stem.split('_')[:2]
      for record in SeqIO.parse(ffile):
        rows.append({'Sequence':record.seq._data, 'SequenceId':record.id, 'Class':class_name, 'SuperClass':super_class_name})
    all_data = pd.DataFrame(data=rows)

    # create train and valid data
    train_df, valid_df = self.create_train_valid_data(all_data)

    # create images for train and valid data
    create_image_dataset(sequences=train_df['Sequence'].tolist(), sequence_ids=train_df['SequenceId'].tolist(), class_name=train_df['Class'], super_class_name=train_df['SuperClass'], save_dir=self.train_data_dir)
    create_image_dataset(sequences=valid_df['Sequence'].tolist(), sequence_ids=valid_df['SequenceId'].tolist(), class_name=valid_df['Class'], super_class_name=valid_df['SuperClass'], save_dir=self.valid_data_dir)

  @staticmethod
  def create_train_valid_data(all_data:pd.DataFrame)->list:
    """
    Method to create train, valid dfs

    Args:
        all_data (pd.DataFrame): [description]

    Returns:
        list: [description]
    """
    train_dfs, valid_dfs = [], []
    for class_name in all_data['Class'].unique():
      class_df = all_data[all_data['Class']==class_name].sample(n=1000, random_state=32)
      num_train_samples = int(round(class_df.shape[0] * 0.7))
      train_dfs.append(class_df.iloc[:num_train_samples,:])
      valid_dfs.append(class_df.iloc[num_train_samples:,:])
    return pd.concat(train_dfs,axis='rows'), pd.concat(valid_dfs, axis='rows')

  @staticmethod
  def check_dir_exists(dir_pth:Path)->None:
    """Method to check if a dir exists, if not create one

    Args:
        dir_pth (Path): [description]
    """
    if not dir_pth.exists():
      dir_pth.mkdir(parents=True, exist_ok=True)

  def train(self):
    """Method to train model
    """
    # create data bunch
    data = ImageDataBunch.from_folder(self.clsfn_data, train=self.train_data_dir, valid=self.valid_data_dir, ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
    learn = create_cnn(data, models.resnet34, metrics=accuracy)
    defaults.device = torch.device('cuda') # makes sure the gpu is used
    learn.fit_one_cycle(4, max_lr=slice(3e-5, 3e-4))
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()

