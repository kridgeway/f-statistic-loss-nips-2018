import os
import numpy as np
from matplotlib import pyplot as plt
import smallnorb_dataset
import sys

# Borrowed from https://github.com/ndrplz/small_norb
# Download dataset files:
#   https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz
#   https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz
#   https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz
#   https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz
#   https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz
#   https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz
# Uncompress into a single folder
ds = smallnorb_dataset.SmallNORBDataset(dataset_root=sys.argv[1])
train = ds.data['train']
test = ds.data['test']
ex = ds.data['train']
ns = len(train) + len(test)

images = np.zeros( (ns, 2, 96, 96), dtype=np.uint8 )
idx=0
categories=[]
instances=[]
elevations=[]
azimuths=[]
lightings=[]
poses=[]
traintests=[]
def append(ds,traintest,image_idx):
    for tidx in xrange(len(ds)):
        ex = ds[tidx]
        l = ex.image_lt
        r = ex.image_rt
        images[image_idx,0,:,:] = l
        images[image_idx,1,:,:] = r
        categories.append(ex.category)
        instances.append(ex.instance)
        elevations.append(ex.elevation)
        azimuths.append(ex.azimuth)
        traintests.append(traintest)
        lightings.append(ex.lighting)
        poses.append(ex.pose)
        image_idx+=1
    return image_idx
idx =append(train,0,idx)
idx =append(test,1,idx)

np.savez(os.path.dirname(__file__) + '/small_norb.npz',
    imagedata= images,
    category= np.array(categories, dtype=np.uint8),
    instance= np.array(instances),
    elevation=np.array(elevations, dtype=np.uint8),
    azimuth=  np.array(azimuths, dtype=np.uint8),
    traintest=np.array(traintests, dtype=np.uint8),
    lighting=np.array(lightings, dtype=np.int32),
    pose= np.array(poses, dtype=np.int32),
)
