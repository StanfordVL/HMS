import pickle as pkl
import os
import numpy as np
from tempfile import TemporaryFile
from glob import glob
from pathlib import Path

PICKLE_PATH = ['data/containers/shelves/normal/placements/'] #"data/containers/fridges/12252/placements/"]
# Load the output files with np.load(file)
# More info here: https://numpy.org/doc/stable/reference/generated/numpy.save.html

# Output directory
if not os.path.isdir('./imgs_3d/'):
  os.mkdir('./imgs_3d/')

path = './imgs_3d'

# example: data/containers/fridges/12252/placements/shelf_0/Fruit/shelf_setup_4.pkl
for placement in PICKLE_PATH:
  shelves = os.listdir(placement)
  for shelf in shelves:
    objects = os.listdir(os.path.join(placement, shelf))
    for obj in objects:
      #path = os.path.join(os.path.join(placement, shelf), obj)
      files = glob(os.path.join(os.path.join(placement, shelf), obj) + '/*.pkl')
      for file in files:
        out_path = os.path.join(os.path.join(os.path.join('./imgs_3d/', shelf), obj), file.split('/')[-1].split('.')[0])
        with open(file, 'rb') as out:
          p = pkl.load(out)
          keys = p.keys()
          for key in keys:
            path = os.path.join(out_path, str(key))
            #print(f'{p[key].keys()}')
            if 'im3d' not in p[key].keys():
              continue
            # outfile = TemporaryFile()
            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_im3d'
            print(f'{title}')
            print(f'{path}')
            Path(path).mkdir(parents=True, exist_ok=True)
            np.save(path+'/'+title, p[key]['im3d'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_depth_im3d'
            np.save(path+'/'+title, p[key]['depth_im3d'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_segmask'
            np.save(path+'/'+title, p[key]['segmask'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_camera_intrinsics'
            np.save(path+'/'+title, p[key]['K'])

