from __future__ import absolute_import

import numpy as np
import io
import cv2
cv2.ocl.setUseOpenCL(False)
from PIL import Image
import random

def pil_loader(img_str):
    # print(img_str)
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    open_cv_image = np.array(img)
    return open_cv_image

def put_gaussian_map(label_h, label_w, kpt, sigma=1.0):
    size = 2*3*sigma + 1 
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    radius = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) 

    ret = np.zeros((label_h, label_w), dtype=np.float32)
    if kpt[0]<0 or kpt[0]>=label_w or kpt[1]<0 or kpt[1]>=label_h:
        return ret 

    l = max(0, kpt[0] - radius)
    t = max(0, kpt[1] - radius)
    r = min(label_w-1, kpt[0] + radius)
    b = min(label_h-1, kpt[1] + radius)

    ml = x0 - min(kpt[0], radius)
    mt = y0 - min(kpt[1], radius)
    mr = x0 + min(label_w-1-kpt[0], radius)
    mb = y0 + min(label_h-1-kpt[1], radius)
    l,t,r,b = list(map(int, [l,t,r,b]))
    ml,mt,mr,mb = list(map(int, [ml,mt,mr,mb]))
    ret[t:b+1, l:r+1] = g[mt:mb+1, ml:mr+1]
    return ret


# def crop(image, ctr, box_w, box_h, rot, in_w, in_h):
#     pad_w, pad_h = int(box_w/2), int(box_h/2)
#
#     # pad image
#     pad_mat = cv2.copyMakeBorder(image,
#                                 top=pad_h,
#                                 bottom=pad_h,
#                                 left=pad_w,
#                                 right=pad_w,
#                                 borderType= cv2.BORDER_CONSTANT,
#                                 value=[0,0,0])
#
#     l = ctr[0] - box_w/2 + pad_w
#     t = ctr[1] - box_h/2 + pad_h
#     r = ctr[0] + box_w/2 + pad_w
#     b = ctr[1] + box_h/2 + pad_h
#     l,t,r,b = map(int, [l,t,r,b])
#     image_roi = pad_mat[t:b, l:r, :]
#     image_roi = cv2.resize(image_roi, (in_w, in_h))
#
#     # rotate
#     rows, cols, channels = image_roi.shape
#     assert channels == 3
#     pad_ctr = (int(cols/2), int(rows/2))
#     rot_matrix = cv2.getRotationMatrix2D(pad_ctr, rot, 1)
#     ret = cv2.warpAffine(image_roi, rot_matrix, (in_w, in_h))
#     return ret

def crop(image, ctr, box_w, box_h, rot, in_w, in_h):
    pad_w, pad_h = int(box_w/2), int(box_h/2)

    # pad image
    pad_mat = cv2.copyMakeBorder(image,
                                top=pad_h,
                                bottom=pad_h,
                                left=pad_w,
                                right=pad_w,
                                borderType= cv2.BORDER_CONSTANT,
                                value=[0,0,0])

    l = ctr[0] - box_w/2 + pad_w
    t = ctr[1] - box_h/2 + pad_h
    r = ctr[0] + box_w/2 + pad_w
    b = ctr[1] + box_h/2 + pad_h
    l,t,r,b = map(int, [l,t,r,b])
    image_roi = pad_mat[t:b, l:r, :]

    # h = random.randint(52, 65)
    # w = int((h * box_w) / box_h)
    # img2 = cv2.resize(image_roi, (w, h))
    # image_roi = cv2.resize(img2, (in_w, in_h))

    image_roi = cv2.resize(image_roi, (in_w, in_h))

    # rotate
    rows, cols, channels = image_roi.shape
    assert channels == 3
    pad_ctr = (int(cols/2), int(rows/2))
    rot_matrix = cv2.getRotationMatrix2D(pad_ctr, rot, 1)
    ret = cv2.warpAffine(image_roi, rot_matrix, (in_w, in_h))
    return ret

def get_transform(center, w, h, rot, res_w, res_h):
    """
    General image processing functions
    """
    # Generate transformation matrix
    scale_w = float(res_w) / w
    scale_h = float(res_h) / h
    t = np.zeros((3, 3))
    t[0, 0] = scale_w
    t[1, 1] = scale_h
    t[2, 2] = 1
    t[0, 2] = -float(center[0]) * scale_w + .5 * res_w
    t[1, 2] = -float(center[1]) * scale_h + .5 * res_h
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res_w/2
        t_mat[1,2] = -res_h/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def kpt_transform(pts, center, box_w, box_h, rot, res_w, res_h, invert):
    t = get_transform(center, box_w, box_h, rot, res_w, res_h)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pts[0], pts[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)


def tensor_to_image(inputs, idx):
    img = np.squeeze(inputs.cpu().numpy()[idx,:,:,:]).copy()
    img = np.transpose(img, [1,2,0])
    img = img * np.array([256,256,256]) + np.array([128,128,128])
    return img.astype(np.uint8).copy()

def numpy_to_image(inputs, idx):
    img = np.squeeze(inputs[idx,:,:,:]).copy()
    img = np.transpose(img, [1,2,0])
    img = img * np.array([256,256,256]) + np.array([128,128,128])
    return img.astype(np.uint8).copy()

