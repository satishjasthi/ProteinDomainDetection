import os, requests
import pandas as pd
from string import Template
from pathlib import Path

ProjectRoot = Path(__file__).resolve().parent.parent.parent

# catalytic all classes full data (seed + non seed data)
catalytic_full_data = pd.read_csv(ProjectRoot/'data/PfamData/Catalytic_ModelData_full.csv')

# Download Pfam Full sequence data, these classes are taken from Pfam Full data for catalytic and binding domain data
class_subclass_map = {'Lysozyme':[x.replace('_', '-') for x in catalytic_full_data[catalytic_full_data['Class']=='Lysozyme']['SubClass'].unique()],
                      'peptidase':[x.replace('_',  '-') for x in catalytic_full_data[catalytic_full_data['Class']=='peptidase']['SubClass'].value_counts().keys()[:10]],
                    #   'Amidase_2':['Amidase-PF01510'],
                    #   'Amidase_3':['Amidase-PF01520'],
                    #   'CHAP': ['CHAP-PF05257'],
                    #   'SH3_4': ['SH34-PF06347'],
                    #   'SH3_3': ['SH33-PF08239'],
                    #   'SH3_5': ['SH35-PF08460'],
                    #   'LysM': ['LysM-PF01476']
                    
                      }

for class_name, subclass_list in class_subclass_map.items():
    for subclass in subclass_list:
        
        class_dir = ProjectRoot/f'data/PfamData/{class_name}___full_sequence_data'
        if not class_dir.exists():
            class_dir.mkdir(parents=True, exist_ok=True)
        subclass_id = subclass.split('-')[-1]
        file_path = class_dir/f'{class_name}-{subclass_id}___full_sequence_data.fasta.gz'
        dom_file_path = class_dir/f'{class_name}-{subclass_id}___domain_data.fasta'
        os.system(f"wget -O {str(file_path)} https://pfam.xfam.org/family/{subclass_id}/alignment/long/gzipped")
        # download full data to get the just domain data 
        url = Template('https://pfam.xfam.org/family/alignment/download/format?acc=$accessionId&alnType=full&format=fasta&order=t&case=u&gaps=none&download=1')
        content = requests.get(url.substitute(accessionId=subclass_id)).text
        with open(dom_file_path,'w') as handle:
            handle.write(content)        


