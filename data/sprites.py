import sh
import numpy as np
import h5py
import re
import sys
import os

images=[]
identities=[]
poses=[]
frame_ids=[]
labels=[]

for sprite_fn in sh.find(sys.argv[1],'-iname','*.mat'):
    sprite_fn = sprite_fn.strip()
    result = re.search(r"sprites_(\d+).mat",sprite_fn)
    if result:
        identity_str = str(result.groups()[0])
        identity = int(identity_str)
        with h5py.File(sprite_fn,'r') as f:
            print 'loaded', sprite_fn
            n_poses = f['sprites'].shape[0]
            f_labels= np.array(f['labels'])
            for pose_idx in xrange(n_poses):
                pose_data = f[f['sprites'][pose_idx][0]]
                n_frames = pose_data.shape[0]
                for frame_idx in xrange(n_frames):
                    frame = pose_data[frame_idx].reshape( (3,60,60) ).swapaxes(0,2)
                    images.append(frame)
                    identities.append(identity)
                    poses.append(pose_idx)
                    frame_ids.append(frame_idx)
                    labels.append(f_labels)

labels = np.array(labels)
images_arr = np.zeros((len(images), 60, 60, 3), dtype=np.uint8)
for image_idx in xrange(len(images)):
    images_arr[image_idx] = images[image_idx] * 255.

np.savez(os.path.dirname(__file__) +'/sprites.npz',
        imagedata=images_arr,
        identities=identities,
        poses=poses,
        frame_ids=frame_ids,
        labels=labels)

