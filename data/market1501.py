from PIL import Image
import sys
import glob
import re
import numpy as np
import os

def process_dir(dirname):
    filenames=[]
    cameras=[]
    ids=[]
    for f in glob.glob('%s/%s/*.jpg' % (sys.argv[1],dirname)):
        parts = f.split('/')
        first, camera, identity, imageidx = re.match(r'([^_]*)_(c\ds\d)_([\d]+)_([\d]+).jpg', parts[-1]).groups()
        if first != '-1' and first != '0000':
            filenames.append(f)
            cameras.append(camera)
            ids.append(first)
        else:
            print f
    print 'unique ids', np.unique(np.array(ids)).shape
    images = np.zeros((len(ids), 128, 64, 3), dtype=np.uint8)
    for idx in xrange(len(ids)):
        img = Image.open(filenames[idx])
        arr = np.array(img)
        images[idx] = img
    return images, np.array(ids), np.array(filenames)

test_images, test_ids, test_filenames = process_dir('bounding_box_test')
train_images, train_ids, train_filenames = process_dir('bounding_box_train')

images = np.concatenate( (train_images, test_images) )
ids = np.concatenate( (train_ids, test_ids))
filenames = np.concatenate( (train_filenames, test_filenames) )

pct_train_ids = 0.8
unique_train_ids = np.random.permutation(np.unique(train_ids))
n_train_ids = int(len(unique_train_ids)*0.8)
train_ids = unique_train_ids[:n_train_ids]
val_ids = unique_train_ids[n_train_ids:]
test_ids = np.unique(test_ids)

train_indices_bool = np.in1d(ids, train_ids)
val_indices_bool = np.in1d(ids, val_ids)
test_indices_bool = np.in1d(ids, test_ids)

np.savez(os.path.dirname(__file__) + '/market_1501.npz',
         imagedata = images,
         filename=filenames,
         identity=ids,
         test_indices_bool = test_indices_bool,
         train_indices_bool = train_indices_bool,
         val_indices_bool = val_indices_bool
        )
