import string, random
from PIL import Image
import numpy as np

def generate_amino_acid_color_map()->dict:
  """Function to map every amino acid in a RGB color

  Returns:
      dict: amino_acid:color map
  """
  amino_acid_color_map = {}
  possible_colors = ((255,204,153),
                    (255,180,102),
                    (255, 204,204),
                    (153,255,51),
                    (0,255,0) ,
                    (0,204,102) ,
                    (0,153,153),
                    (0,51,102),
                    (0,0,51 ),
                    (51,0,51) ,
                    (102,0,102),
                    (176,0,153) ,
                    (0,0,204),
                    (0,128,255),
                    (51,255,255),
                    (102,255,178),
                    (153,255, 153),
                    (229,255,204),
                    (102,0,0),
                    (102,90,0), 
                    (0,102,102),
                    (25,0,51) ,
                    (204,0,102) ,
                    (96,96,96) ,
                    (255,51,255),
                    (120,120,120))
  for index, amino_acid in enumerate(string.ascii_uppercase):
      amino_acid_color_map[amino_acid] = possible_colors[index]
  return amino_acid_color_map


def sequence2Image(Sequence:str, height:int=800, width:int=1333, amino_acid_color_map:dict={})->Image:
  """
    Function to convert sequence to Image

  Args:
      Sequence (str): [description]
      height (int, optional): [description]. Defaults to 800.
      width (int, optional): [description]. Defaults to 1333.
      amino_acid_color_map (dict, optional): [description]. Defaults to {}.

  Returns:
      Image: [description]
  """
  # create a blank image
  img_arr = np.zeros((height,width,3)) # numpy takes images in H, W, D order, PIL in (W, H)

  # fill RGB colors to each pixel
  num_rows = img_arr.shape[0]
  num_cols = img_arr.shape[1]
  for row_indx in range(num_rows):
    for column_indx in range(num_cols):
      if column_indx < len(Sequence):
        img_arr[row_indx, column_indx, :] = amino_acid_color_map[Sequence[column_indx]]
      else:
        img_arr[row_indx, column_indx, :] = [255, 255, 255]
  return Image.fromarray(img_arr.astype(np.uint8))


if __name__ == "__main__":
  amino_acid_color_map = generate_amino_acid_color_map()
  sequence2Image("".join([random.choice(string.ascii_uppercase) for i in range(650)]), amino_acid_color_map=amino_acid_color_map).save('img.png')
 