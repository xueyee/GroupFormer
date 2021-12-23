import numpy as np
import skimage.io
import skimage.transform
import pickle
import json

import torch
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models
import torch.nn.functional as F

from PIL import Image
import random

import sys

"""
Reference:
https://github.com/cvlab-epfl/social-scene-understanding/blob/master/volleyball.py
"""

ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
              'l_set', 'l-spike', 'l-pass', 'l_winpoint']

NUM_ACTIVITIES = 8

ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
           'moving', 'setting', 'spiking', 'standing',
           'waiting']
NUM_ACTIONS = 9


def volley_read_annotations(path):
    """
    reading annotations for the given sequence
    """
    annotations = {}

    gact_to_id = {name: i for i, name in enumerate(ACTIVITIES)}
    act_to_id = {name: i for i, name in enumerate(ACTIONS)}

    with open(path) as f:
        for l in f.readlines():
            values = l[:-1].split(' ')
            file_name = values[0]
            activity = gact_to_id[values[1]]

            values = values[2:]
            num_people = len(values) // 5

            action_names = values[4::5]
            actions = [act_to_id[name]
                       for name in action_names]

            def _read_bbox(xywh):
                x, y, w, h = map(int, xywh)
                return y, x, y + h, x + w

            bboxes = np.array([_read_bbox(values[i:i + 4])
                               for i in range(0, 5 * num_people, 5)])

            fid = int(file_name.split('.')[0])
            annotations[fid] = {
                'file_name': file_name,
                'group_activity': activity,
                'actions': actions,
                'bboxes': bboxes,
            }
    return annotations


def volley_read_dataset(path, seqs):
    data = {}
    for sid in seqs:
        data[sid] = volley_read_annotations(path + '/%d/annotations.txt' % sid)
    return data


def volley_all_frames(data):
    frames = []
    for sid, anns in data.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))
    return frames


def volley_random_frames(data, num_frames):
    frames = []
    for sid in np.random.choice(list(data.keys()), num_frames):
        fid = int(np.random.choice(list(data[sid]), []))
        frames.append((sid, fid))
    return frames


def volley_frames_around(frame, num_before=5, num_after=4):
    sid, src_fid = frame
    return [(sid, src_fid, fid)
            for fid in range(src_fid - num_before, src_fid + num_after + 1)]

def volleyball_readpose(data_path):
    f = open(data_path,'r')
    f = f.readlines()
    pose_ann=dict()
    for ann in f:
        ann = json.loads(ann)
        filename=ann['filename'].split('/')
        sid=filename[-3]
        src_id=filename[-2]
        fid=filename[-1][:-4]
        center = [ann['tmp_box'][0], ann['tmp_box'][1]]
        keypoint=[]
        for i in range(0,51,3):
            keypoint.append(ann['keypoints'][i])
            keypoint.append(ann['keypoints'][i+1])
        pose_ann[sid+src_id+fid+str(center)]=keypoint
    return pose_ann

class VolleyballDataset(data.Dataset):
    """
    Characterize volleyball dataset for pytorch
    """

    def __init__(self, config, transform=None):
        self.anns = volley_read_dataset(config.data_path, config.seqs)
        self.frames = volley_all_frames(self.anns)
        self.tracks = pickle.load(open(config.tracks, 'rb'))
        self.images_path = config.data_path
        assert config.sample in ['train', 'val']
        self.sample = config.sample
        assert config.flip in [True, False]
        self.flip = config.flip
        self.transform = transform

        self.pose_anns=volleyball_readpose(config.keypoints)
        
        #root = list(range(len(self.frames)))
        #random.shuffle(root)
        #self.root = root

        self.num_boxes = 12
        self.num_before = 5
        self.num_after = 5



        # self.is_training = is_training
        # self.is_finetune = is_finetune

    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.frames)

    def __getitem__(self, index):
        """
        Generate one sample of the dataset
        """
        #index = self.root[index]
        select_frames = self.volley_frames_sample(self.frames[index])
        sample = self.load_samples_sequence(select_frames)

        return sample

    def volley_frames_sample(self, frame):

        sid, src_fid = frame
        if self.sample == 'train':
            sample_frames = random.sample(range(src_fid - self.num_before, src_fid), 3) + [src_fid] + \
                            random.sample(range(src_fid + 1, src_fid + self.num_after + 1), 3)
            sample_frames.sort()
        elif self.sample == 'val':
            sample_frames = range(src_fid - 3, src_fid + 4, 1)
        else:
            assert False 

        return [(sid, src_fid, fid) for fid in sample_frames]

    def load_samples_sequence(self, select_frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """
        if self.flip and np.random.rand()>0.5:
            flip = True
        else:
            flip = False  
        images, boxes = [], []
        activities, actions = [], []
        poses = []
        flows = []
        for i, (sid, src_fid, fid) in enumerate(select_frames):

            img = Image.open(self.images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
            W,H = img.size
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = self.transform(img)
            # H,W,3 -> 3,H,W
            images.append(img)            

            temp_boxes = np.ones_like(self.tracks[(sid, src_fid)][fid])
            temp_poses = []
            for i, track in enumerate(self.tracks[(sid, src_fid)][fid]):
                y1, x1, y2, x2 = track
                
                X1 = int(round(x1*W))
                Y1 = int(round(y1*H))
                X2 = int(round(x2*W))
                Y2 = int(round(y2*H))

                X1 = min(max(X1,0),W)
                X2 = min(max(X2,0),W)
                Y1 = min(max(Y1,0),H)
                Y2 = min(max(Y2,0),H)
                center = [(X1+X2)/2.,(Y1+Y2)/2.]
                try:
                    keypoint = self.pose_anns[str(sid)+str(src_fid)+str(fid)+str(center)]
                except:
                    try:
                        center[1] -= 0.5
                        keypoint = self.pose_anns[str(sid)+str(src_fid)+str(fid)+str(center)]
                    except:
                        try:
                            center[0] -= 0.5
                            keypoint = self.pose_anns[str(sid)+str(src_fid)+str(fid)+str(center)]
                        except:
                            center[1] += 0.5
                            keypoint = self.pose_anns[str(sid)+str(src_fid)+str(fid)+str(center)]
                size = np.sqrt((X2-X1)*(Y2-Y1)/4)
                keypoint = np.array(keypoint).reshape(17,2)
                center = np.array(center)
                keypoint = (keypoint-center)/size

                if flip:
                    temp_boxes[i] = np.array([1-x2, y1, 1-x1, y2])
                    keypoint[:,0]= keypoint[:,0]*-1.
                    temp_poses.append(keypoint)
                else:
                    temp_boxes[i] = np.array([x1, y1, x2, y2])
                    temp_poses.append(keypoint)
            if len(temp_poses) != self.num_boxes:
                temp_poses = temp_poses + temp_poses[:self.num_boxes -len(temp_poses)]
            temp_poses = np.vstack(temp_poses)
            poses.append(temp_poses)
            boxes.append(temp_boxes)

            #actions.append(self.anns[sid][src_fid]['actions'])

            if len(boxes[-1]) != self.num_boxes:
                boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes - len(boxes[-1])]])
                #actions[-1] = actions[-1] + actions[-1][:self.num_boxes - len(actions[-1])]
            #activities.append(self.anns[sid][src_fid]['group_activity'])
            flow = np.load(self.images_path + '/flow/%d/%d/%d.npy' % (sid, src_fid, fid))
            #flow = np.random.rand(2,180,320)
            flow = torch.from_numpy(flow)
            if flip:
                flow = torch.flip(flow,[2])
                flow[0:1,:,:] = -flow[0:1,:,:]
                flows.append(flow)
            else:
                flows.append(flow)

        actions = self.anns[sid][src_fid]['actions']
        activities = self.anns[sid][src_fid]['group_activity']
        if flip:
            activities = (activities+4)%8
        if len(actions) != self.num_boxes:
            actions = actions + actions[:self.num_boxes - len(actions)]

        images = torch.stack(images)
        flows = torch.stack(flows)
        flows = F.interpolate(flows, size=[180,320], mode='bilinear', align_corners=True)
        activities = np.array(activities, dtype=np.int32)
        bboxes = np.vstack(boxes).reshape([-1, self.num_boxes, 4])
        poses = np.vstack(poses).reshape([-1, 17,2])
        actions = np.hstack(actions).reshape([-1, self.num_boxes])

        # convert to pytorch tensor
        bboxes = torch.from_numpy(bboxes).float()
        poses = torch.from_numpy(poses).float()
        actions = torch.from_numpy(actions).long()
        activities = torch.from_numpy(activities).long()

        return images, bboxes, actions, activities,poses,flows
