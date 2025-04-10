import numpy as np
from PIL import Image, ImageDraw
import math
import csv
import pickle
from skimage.transform import resize
import matplotlib.pyplot as plt
import torch

def load_annotations(csvfile, delimiter=","):
    rows = []
    with open(csvfile, 'r') as file:
        reader = csv.DictReader(file, delimiter=delimiter, quotechar="|")
        for row in reader:
            rows.append(row)
    return rows

def load_og_image(folder, imagename, as_type='RGB', size=224, ext='jpg'):
    img = Image.open(f"{folder}/{imagename}.{ext}")
    img = img.convert(as_type)
    imgnp = np.array(img)
    image = np.zeros([size, size])
    offset = (size - min(imgnp.shape[0], imgnp.shape[1]))//2
    if as_type=="RGB":
        image = np.zeros([size, size, 3])
    else:
        image = np.zeros([size, size])
        if imgnp.shape[0] < imgnp.shape[1]:
            image[offset:offset + imgnp.shape[0], :] = imgnp
        else:
            image[:, offset:offset + imgnp.shape[1]] = imgnp
    return image



def load_image(folder, imagename, bbox1, bbox2, as_type="RGB", size=224):
    img = Image.open(f"{folder}/{imagename}")
    img = img.convert(as_type)
    min_unionbox = np.minimum(bbox1, bbox2)
    max_unionbox = np.maximum(bbox1, bbox2)
    union = [min_unionbox[0], min_unionbox[1], max_unionbox[2], max_unionbox[3]]
    width = union[3] - union[1]
    height = union[2] - union[0]
    ratio = min(width, height) / max(width, height)
    imgnp = np.array(img)
    imgnp = imgnp[union[1]:union[3], union[0]:union[2]]
    if width > height :
        imgnp = resize(imgnp, (size, size * ratio))
    else:
        imgnp = resize(imgnp, (size * ratio, size))
    image = np.zeros([size, size])
    offset = (size - min(imgnp.shape[0], imgnp.shape[1]))//2
    if as_type=="RGB":
        image = np.zeros([size, size, 3])
        if imgnp.shape[0] < imgnp.shape[1]:
            image[offset:offset + imgnp.shape[0], :,:] = imgnp
        else:
            image[:, offset:offset + imgnp.shape[1],:] = imgnp
    else:
        image = np.zeros([size, size])
        if imgnp.shape[0] < imgnp.shape[1]:
            image[offset:offset + imgnp.shape[0], :] = imgnp
        else:
            image[:, offset:offset + imgnp.shape[1]] = imgnp
    return image


def load_image_coco(folder, imagename, bbox1, bbox2, as_type="RGB", size=224):
    img = Image.open(f"{folder}/{imagename}")
    img = img.convert(as_type)
    min_unionbox = np.minimum(bbox1, bbox2)
    max_unionbox = np.maximum(bbox1, bbox2)
    union = [min_unionbox[0], min_unionbox[1], max_unionbox[2], max_unionbox[3]]
    width = union[1] - union[0]
    height = union[3] - union[2]
    ratio = min(width, height) / max(width, height)
    imgnp = np.array(img)
    imgnp = imgnp[union[0]:union[1], union[2]:union[3]]
    if width > height :
        imgnp = resize(imgnp, (size, size * ratio))
    else:
        imgnp = resize(imgnp, (size * ratio, size))
    image = np.zeros([size, size])
    offset = (size - min(imgnp.shape[0], imgnp.shape[1]))//2
    if as_type=="RGB":
        image = np.zeros([size, size, 3])
        if imgnp.shape[0] < imgnp.shape[1]:
            image[offset:offset + imgnp.shape[0], :,:] = imgnp
        else:
            image[:, offset:offset + imgnp.shape[1],:] = imgnp
    else:
        image = np.zeros([size, size])
        if imgnp.shape[0] < imgnp.shape[1]:
            image[offset:offset + imgnp.shape[0], :] = imgnp
        else:
            image[:, offset:offset + imgnp.shape[1]] = imgnp
    return image


def load_fbanner(folder, imagename, size=224, channels="L", ext="png"):
    name = imagename.split(".")[0]
    img = Image.open(f"{folder}/{name}.{ext}")
    if size != 0:
        img = img.resize((size, size), resample=Image.Resampling.BILINEAR)
    img = img.convert(channels)
    return np.array(img)


def load_image_bounding_box(folder, imagename, bbox1, bbox2, size=224):
    name = imagename.split(".")[0]
    img = Image.open(f"{folder}/{name}.jpg")
    if size != 0:
        img = img.resize((size, size), resample=Image.Resampling.BILINEAR)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox1)
    draw.rectangle(bbox2)
    return np.array(img)



def crop_image(image, bg, v1, v2):
    x0, y0 = np.where(image == v1)
    x1, y1 = np.where(image == v2)

    xmin = min(np.min(x0), np.min(x1))
    xmax = max(np.max(x0), np.max(x1))
    ymin = min(np.min(y0), np.min(y1))
    ymax = max(np.max(y0), np.max(y1))
    
    x_size = xmax - xmin
    y_size = ymax - ymin
    max_size = max(x_size, y_size)

    x_offset = int((max_size - x_size)/2)
    y_offset = int((max_size - y_size)/2)

    new_image = np.ones((max_size, max_size)) * bg
    new_image[x_offset:x_offset+x_size, y_offset:y_offset+y_size] = image[xmin:xmax, ymin:ymax]
    return new_image


def bounding_box_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(rmin), int(rmax), int(cmin), int(cmax)

def bounding_box(image, v1=254, v2=255):
    rmin, rmax, cmin, cmax = bounding_box_mask(image == v1)
    x1, y1, w1, h1 = rmin, cmin, rmax-rmin, cmax-cmin
    rmin, rmax, cmin, cmax = bounding_box_mask(image == v2)
    x2, y2, w2, h2 = rmin, cmin, rmax-rmin, cmax-cmin
    return (x1, y1, w1, h1), (x2, y2, w2, h2)


def vtranse_bboxes(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    tx1 = (x1 - x2)/w2
    ty1 = (y1 - y2)/h2
    tw1 = math.log(w1/w2)
    th1 = math.log(h1/h2)
    tx2 = (x2 - x1)/w1
    ty2 = (y2 - y1)/h1
    tw2 = math.log(w2/w1)
    th2 = math.log(h2/h1)
    return np.array([tx1, ty1, tw1, th1, tx2, ty2, tw2, th2])


def bounding_box_2d(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return np.array([x1, y1, x1+w1, y1+h1, x2, y2, x2+w2, y2+h2])


def mask_bbox(box1, box2, size=224):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    im1 = np.zeros((size,size))
    im2 = np.zeros((size,size))
    im1[x1:x1+w1, y1:y1+h1] = 1
    im2[x2:x2+w2, y2:y2+h2] = 1
    return im1, im2


def save_pickle(path, obj): # path.pkl
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def load_image_2b(folder, imagename, as_type="L"):
    size = 112
    img = Image.open(f"{folder}/{imagename}")
    img = img.resize((size, size), Image.Resampling.NEAREST)
    img = img.convert(as_type)
    img = np.array(img)
    colors = set(np.unique(img))
    colors.remove(np.min(img))
    objects = []
    for n, c in enumerate(colors):
        img_obj = np.zeros_like(img)
        img_obj[img == c] = 1
        objects.append(img_obj)
    objects.append(img)
    return objects


def reshape_image(images):
    if len(images.shape) == 3:
        image = images.unsqueeze(1)
    else: 
        image = torch.permute(images, (0, 3, 1, 2))
    return image