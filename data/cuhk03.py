from PIL import Image
import os
import glob
import re
import numpy as np
import sys

filenames=[]
campairs=[]
ids=[]

#Download CUHK03 http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html

for f in glob.glob('{}/labeled_normalized/**/*.png'.format(sys.argv[1])):
    parts = f.split('/')
    pair, identity = re.search(r'campair(\d)_id([\d]+)',parts[-2]).groups()
    campairs.append(int(pair))
    ids.append(int(identity))
    filenames.append(f)

data = np.zeros( (len(ids), 160, 60, 3), dtype=np.uint8 )
for idx in xrange(len(ids)):
    img = Image.open(filenames[idx])
    arr = np.array(img)
    data[idx] = arr

combined_id = np.array(campairs) * 10000 + np.array(ids)

test_camera_id = np.load('{}/test_camera_id.npy'.format(sys.argv[1]))
test_combined_id = test_camera_id[0,:] * 10000 + test_camera_id[1,:]
test_indices_bool = np.in1d(combined_id, test_combined_id)

remaining_ids = np.setdiff1d(combined_id, test_combined_id)
valid_ids = np.random.choice(remaining_ids, 100, replace=False)
valid_indices_bool = np.in1d(combined_id, valid_ids)

np.savez(os.path.dirname(__file__) + '/cuhk_dataset.npz',
         imagedata=data,
         filename=filenames,
         campair=campairs,
         original_id=ids,
         identity=combined_id,
         test_indices_bool=test_indices_bool,
         valid_indices_bool=valid_indices_bool
         )
