import pickle as pkl
import os
import numpy as np
from tempfile import TemporaryFile
from glob import glob

PICKLE_PATH = ['data/containers/shelves/normal/placements/', "data/containers/fridges/12252/placements/"]
# Load the output files with np.load(file)
# More info here: https://numpy.org/doc/stable/reference/generated/numpy.save.html

# Output directory
if not os.path.isdir('./imgs_3d/'):
  os.mkdir('./imgs_3d/')

# example: data/containers/fridges/12252/placements/shelf_0/Fruit/shelf_setup_4.pkl
for placement in PICKLE_PATH:
  shelves = os.listdir(placement)
  for shelf in shelves:
    objects = os.listdir(os.path.join(PICKLE_PATH, shelf))
    for obj in objects:
      files = glob(os.path.join(os.path.join(PICKLE_PATH, shelf), obj) + '*.pkl')
      for file in files:
        with open(file, 'rb') as out:
          p = pkl.load(out)
          keys = p.keys()
          for key in keys:
        		# outfile = TemporaryFile()
            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_im3d'
            np.save(title, p[key]['im3d'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_depth_m3d'
            np.save('./imgs_3d/'+title, p[key]['depth_im3d'])