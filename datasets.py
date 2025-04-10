import torch
from tqdm import tqdm
import random
import torchvision
import json
from image import *
from evaluation import spatial_label_fusion

class SpatialSense(torch.utils.data.Dataset):
    def __init__(self, ant, splitname, fbanners, gauss=""):
        super().__init__()
        self.imgpath = "../Datasets/*"
        self.fbannerpath = "../Datasets/*"
        self.imgpath_flip = "../Datasets/*"
        self.fbannerpath_flip = "../Datasets/*"
        self.annotations = []

        for i in tqdm(range(len(ant))):
            if ant[i]['split'] == splitname:
                imagename = ant[i]["imagename"].split('.')[0]
                bs = ant[i]['annotations']['subject']['bbox']
                bo = ant[i]['annotations']['object']['bbox']
                bbox1 = [bs[2], bs[0], bs[3], bs[1]]
                bbox2 = [bo[2], bo[0], bo[3], bo[1]]
                bboxes = bbox1 + bbox2
                image = load_image(self.imgpath, f"{imagename}.jpg", bbox1, bbox2).astype(np.float32, copy=False)
                fb = load_fbanner(self.fbannerpath, f"{imagename}.png", channels="L")
                bin_img_0 = load_fbanner("../Datasets/*", imagename + "_0", 224, "L", "png").astype(np.float32, copy=False)
                bin_img_1 = load_fbanner("../Datasets/*", imagename + "_1", 224, "L", "png").astype(np.float32, copy=False)
                fb_pca = fbanners[imagename]
                entity = {
                    "annotations": ant[i],
                    "imagename": imagename,
                    "input": image,
                    "fbanner": fb.astype(np.float32, copy=False),
                    "fbanner_pca": np.array(fb_pca).astype(np.float32, copy=False),
                    "rel": spatial_label_fusion(ant[i]['annotations']['predicate']),
                    "subj": ant[i]["annotations"]["subject"]["name"],
                    "obj": ant[i]["annotations"]["object"]["name"],
                    "bboxes": np.array(bboxes).astype(np.float32, copy=False),
                    "bin_img": [bin_img_0, bin_img_1]
                }
                self.annotations.append(entity)
                if splitname != "test":
                    image = load_image(self.imgpath_flip, f"{imagename}.jpg", bbox1, bbox2).astype(np.float32, copy=False)
                    fb = load_fbanner(self.fbannerpath_flip, f"{imagename}.png", channels="L").flatten()
                    entity = {
                        "annotations": ant[i],
                        "imagename": imagename,
                        "input": image,
                        "fbanner": fb.astype(np.float32, copy=False),
                        "fbanner_pca": np.array(fb_pca).astype(np.float32, copy=False),
                        "rel": spatial_label_fusion(ant[i]['annotations']['predicate']),
                        "subj": ant[i]["annotations"]["subject"]["name"],
                        "obj": ant[i]["annotations"]["object"]["name"],
                        "bboxes": np.array(bboxes).astype(np.float32, copy=False)
                    }
                    self.annotations.append(entity)

    def __getitem__(self, idx):
        itm = self.annotations[idx]
        return itm


    def __len__(self):
        return len(self.annotations)
    


class Unrel(torch.utils.data.Dataset):
    def __init__(self, ant, imgpath, fbannerpath):
        super().__init__()
        self.imgpath = imgpath
        self.fbannerpath = fbannerpath
        self.annotations = []
        for i in tqdm(range(len(ant))):
            imagename = ant[i]["imagename"]
            bbox1 = ant[i]["annotations"]["obj"]["bbox"]
            bbox2 = ant[i]["annotations"]["subj"]["bbox"]
            bboxes = np.array(bbox1 + bbox2)
            image = load_image(self.imgpath, imagename, bbox1, bbox2, as_type="RGB").astype(np.float32, copy=False)
            fb = load_fbanner(self.fbannerpath, imagename).astype(np.float32, copy=False)
            bin_img_0 = load_fbanner("../Datasets/*", ant[i]["imagename"].split('.')[0]+"_0", 224, "L", "png").astype(np.float32, copy=False)
            bin_img_1 = load_fbanner("../Datasets/*", ant[i]["imagename"].split('.')[0]+"_1", 224, "L", "png").astype(np.float32, copy=False)
            entity = {
                "imagename": imagename,
                "annotations": ant[i],
                "input": image,
                "fbanner": fb,
                "rel": ant[i]["annotations"]["predicate"],
                "subj": ant[i]["annotations"]["subj"]["name"],
                "obj": ant[i]["annotations"]["obj"]["name"],
                "bboxes":bboxes.astype(np.float32, copy=False),
                "bin_img": np.dstack((bin_img_0, bin_img_1))
            }
            self.annotations.append(entity)

    def __getitem__(self, idx):
        itm = self.annotations[idx]
        return itm

    def __len__(self):
        return len(self.annotations)
    

class COCO2(torch.utils.data.Dataset):
    def __init__(self, ant, indexes, imgpath, fbannerpath):
        super().__init__()
        self.imgpath = imgpath
        self.fbannerpath = fbannerpath
        self.annotations = []
        for i in tqdm(range(len(indexes))):
            imagename = ant[i]["imagename"]
            bbox1 = ant[i]['bbox1']
            bbox2 = ant[i]['bbox2']
            image = load_image_coco(self.imgpath, imagename + ".jpg", bbox1, bbox2, "RGB", 224).astype(np.float32, copy=False)
            fbanner = load_fbanner(self.fbannerpath, imagename, 224, "L", "png").astype(np.float32, copy=False)
            entity = {
                "imagename": imagename,
                "input": image,
                "fbanner": fbanner,
                "bboxes": np.array(bbox1 + bbox2).astype(np.float32, copy=False)
            }
            self.annotations.append(entity)
    
    def __getitem__(self, idx):
        itm = self.annotations[idx]
        return itm

    def __len__(self):
        return len(self.annotations)
    


class SenseCOCO(torch.utils.data.Dataset):
    def __init__(self, splitname):
        super().__init__()
        coco_root_folder = "../Datasets/COCO"
        coco_dataset_folder = coco_root_folder + "/train2017"
        coco_annotation_file = coco_root_folder + "/*.json"
        coco_annotation_file_3 = coco_root_folder + "/*.json"
        coco_fbanner_folder = coco_root_folder + "/*"
        coco_fbanner_folder_3 = coco_root_folder + "/*"
        coco_annotations = json.load(open(coco_annotation_file))[:]
        coco_annotations_3 = json.load(open(coco_annotation_file_3))[:]

        sense_root_folder = "../Datasets/SpatialSense"
        sense_dataset_folder = sense_root_folder + "*"
        sense_annotation_file = sense_root_folder + "/*.json"
        sense_fbanner_folder = sense_root_folder + "/*"
        sense_annotations = json.load(open(sense_annotation_file))[:]

        self.annotations = []
        indexes = list(range(len(coco_annotations)))
        train_indexes = indexes[:int(.8*len(indexes))]
        valid_indexes = indexes[int(.8*len(indexes)):]
        if splitname == "train":
            indices = train_indexes
        else:
            indices = valid_indexes
        for i in tqdm(range(len(coco_annotations[:]))):
            if i in indices:
                imagename = coco_annotations[i]["imagename"]
                bbox1 = coco_annotations[i]['bbox1']
                bbox2 = coco_annotations[i]['bbox2']
                image = load_image_coco(coco_dataset_folder, imagename + ".jpg", bbox1, bbox2, "RGB", 224).astype(np.float32, copy=False)
                fbanner = load_fbanner(coco_fbanner_folder, imagename, 224, "L", "png").astype(np.float32, copy=False)
                bin_img_0 = load_fbanner("../Datasets/COCO/*", imagename + "_0", 224, "L", "png").astype(np.float32, copy=False)
                bin_img_1 = load_fbanner("../Datasets/COCO/*", imagename + "_1", 224, "L", "png").astype(np.float32, copy=False)
                entity = {
                    "imagename": imagename,
                    "input": image,
                    "fbanner": fbanner,
                    "bboxes": np.array(bbox1 + bbox2).astype(np.float32, copy=False),
                    "bin_img": np.dstack((bin_img_0, bin_img_1))
                }
                self.annotations.append(entity)
        print(len(self.annotations))

        indexes = list(range(len(coco_annotations_3)))
        train_indexes = indexes[:int(.8*len(indexes))]
        valid_indexes = indexes[int(.8*len(indexes)):]
        if splitname == "train":
            indices = train_indexes
        else:
            indices = valid_indexes
        for i in tqdm(range(len(coco_annotations_3[:]))):
            if i in indices:
                imagename = coco_annotations_3[i]["imagename"]
                bbox1 = coco_annotations_3[i]['bbox1']
                bbox2 = coco_annotations_3[i]['bbox2']
                image = load_image_coco(coco_dataset_folder, imagename + ".jpg", bbox1, bbox2, "RGB", 224).astype(np.float32, copy=False)
                fbanner = load_fbanner(coco_fbanner_folder_3, imagename, 224, "L", "png").astype(np.float32, copy=False)
                bin_img_0 = load_fbanner("../Datasets/COCO/*", imagename + "_0", 224, "L", "png").astype(np.float32, copy=False)
                bin_img_1 = load_fbanner("../Datasets/COCO/*", imagename + "_1", 224, "L", "png").astype(np.float32, copy=False)
                entity = {
                    "imagename": imagename,
                    "input": image,
                    "fbanner": fbanner,
                    "bboxes": np.array(bbox1 + bbox2).astype(np.float32, copy=False),
                    "bin_img": np.dstack((bin_img_0, bin_img_1))
                }
                self.annotations.append(entity)
        print(len(self.annotations))

        for i in tqdm(range(len(sense_annotations[:]))):
            if sense_annotations[i]['split'] == splitname:
                imagename = sense_annotations[i]["imagename"].split('.')[0]
                bs = sense_annotations[i]['annotations']['subject']['bbox']
                bo = sense_annotations[i]['annotations']['object']['bbox']
                bbox1 = [bs[2], bs[0], bs[3], bs[1]]
                bbox2 = [bo[2], bo[0], bo[3], bo[1]]
                image = load_image(sense_dataset_folder, f"{imagename}.jpg", bbox1, bbox2).astype(np.float32, copy=False)
                fb = load_fbanner(sense_fbanner_folder, f"{imagename}.png", channels="L").astype(np.float32, copy=False)
                bin_img_0 = load_fbanner("../Datasets/SpatialSense/*", imagename + "_0", 224, "L", "png").astype(np.float32, copy=False)
                bin_img_1 = load_fbanner("../Datasets/SpatialSense/*", imagename + "_1", 224, "L", "png").astype(np.float32, copy=False)
                entity = {
                        "imagename": imagename,
                        "input": image,
                        "fbanner": fb,
                        "bboxes": np.array(bbox1 + bbox2).astype(np.float32, copy=False),
                        "bin_img": np.dstack((bin_img_0, bin_img_1))
                    }
            self.annotations.append(entity)

    def __getitem__(self, idx):
        itm = self.annotations[idx]
        return itm

    def __len__(self):
        return len(self.annotations)
    


class GQA_rel(torch.utils.data.Dataset):
    def __init__(self, ants, keys, imgpath):
        super().__init__()
        self.annotations = []
        for i in tqdm(range(len(keys))):
            el = ants[keys[i]]
            question = el['question']
            answer = el['answer']
            image_id = el['imageId']
            if answer == "yes":
                onehot_answer = np.array([1.0, 0.0]).astype(np.float32)
            else:
                onehot_answer = np.array([0.0, 1.0]).astype(np.float32)
            image = load_og_image(imgpath, image_id, as_type="RGB", size=224).astype(np.float32, copy=False)
            obj = {
                'question': question,
                'image': image,
                'answer': onehot_answer
            }
            self.annotations.append(obj)


    def __getitem__(self, idx):
        itm = self.annotations[idx]
        return itm

    def __len__(self):
        return len(self.annotations)