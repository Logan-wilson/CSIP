import torch
from tqdm import tqdm
from models import *
import numpy as np
from image import *
import json
from evaluation import *
from baseline_COCO import *


class model_vrd(torch.nn.Module):
    def __init__(self, model, embedding_size, n_classes=9):
        super(model_vrd, self).__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

        self.fc1 = torch.nn.Linear(embedding_size, embedding_size)
        self.batchnorm1 = torch.nn.BatchNorm1d(embedding_size)
        self.fc2 = torch.nn.Linear(embedding_size, embedding_size)
        self.batchnorm2 = torch.nn.BatchNorm1d(embedding_size)
        self.final_fc = torch.nn.Linear(embedding_size, n_classes)
        self.relu = torch.nn.ReLU()


    def forward(self, image, bboxes, bin_img):
        _, embedding = self.model(image, bin_img)
        x1 = self.batchnorm1(self.relu(self.fc1(embedding)))
        x2 = self.batchnorm2(self.relu(self.fc2(x1)))
        x3 = self.relu(self.final_fc(x2))
        return x3


class SpatialSense_VRD(torch.utils.data.Dataset):
    def __init__(self, ant, splitname):
        super().__init__()
        self.imgpath = "../Dataset/path"
        self.imgpath_flip = "../Dataset/path"
        self.annotations = []
        self.predicates = ["above", "behind", "in", "in front of", "next to", "on", "to the left of", "to the right of", "under"]

        for i in tqdm(range(len(ant))):
            if ant[i]['split'] == splitname:
                imagename = ant[i]["imagename"].split('.')[0]
                bs = ant[i]['annotations']['subject']['bbox']
                bo = ant[i]['annotations']['object']['bbox']
                bbox1 = [bs[2], bs[0], bs[3], bs[1]]
                bbox2 = [bo[2], bo[0], bo[3], bo[1]]
                bboxes = bbox1 + bbox2
                image = load_image(self.imgpath, f"{imagename}.jpg", bbox1, bbox2).astype(np.float32, copy=False)
                bin_img_0 = load_fbanner("../Datasets/SpatialSense/segmentations/image_bbox", imagename + "_0", 224, "L", "png").astype(np.float32, copy=False)
                bin_img_1 = load_fbanner("../Datasets/SpatialSense/segmentations/image_bbox", imagename + "_1", 224, "L", "png").astype(np.float32, copy=False)
                entity = {
                    "annotations": ant[i],
                    "imagename": imagename,
                    "input": image,
                    "rel": ant[i]['annotations']['predicate'],
                    "subj": ant[i]["annotations"]["subject"]["name"],
                    "obj": ant[i]["annotations"]["object"]["name"],
                    "bboxes": np.array(bboxes).astype(np.float32, copy=False),
                    "bin_img": np.dstack((bin_img_0, bin_img_1))
                }
                self.annotations.append(entity)
                if splitname != "test":
                    image = load_image(self.imgpath_flip, f"{imagename}.jpg", bbox1, bbox2).astype(np.float32, copy=False)
                    entity = {
                        "annotations": ant[i],
                        "imagename": imagename,
                        "input": image,
                        "rel": ant[i]['annotations']['predicate'],
                        "subj": ant[i]["annotations"]["subject"]["name"],
                        "obj": ant[i]["annotations"]["object"]["name"],
                        "bboxes": np.array(bboxes).astype(np.float32, copy=False),
                        "bin_img": np.dstack((bin_img_0, bin_img_1))
                    }
                    self.annotations.append(entity)

    def __getitem__(self, idx):
        itm = self.annotations[idx]
        pred_rel = np.zeros((9,), dtype=np.float32)
        pred_rel[self.predicates.index(itm["rel"])] = 1.0
        itm['pred_rel'] = pred_rel
        return itm

    def __len__(self):
        return len(self.annotations)



def finetune_model(model, loader, criterion, optimizer):
    model.train()
    losses = []
    acc = 0.0
    num_samples = 0
    for idx, batch in enumerate(tqdm(loader)):
        images = batch["input"]
        bboxes = batch["bboxes"]
        predi = batch["pred_rel"]
        bin_img = batch["bin_img"]
        images = reshape_image(images)
        bin_img = reshape_image(bin_img)
        output = model(images, bboxes, bin_img)
        loss_batch = criterion(output, predi)
        loss_batch1 = loss_batch.item()
        loss = len(batch["input"]) * loss_batch1
        acc += num_true_positives(output, predi)
        num_samples += len(batch["input"])
        losses.append(loss/len(batch["input"]))
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
    losses = np.mean(losses)
    acc /= num_samples
    return loss, acc


def valid_finetuning(model, loader, criterion):
    model.eval()
    losses = []
    acc = 0.0
    num_samples = 0
    for idx, batch in enumerate(tqdm(loader)):
        images = batch["input"]
        bboxes = batch["bboxes"]
        predi = batch["pred_rel"]
        bin_img = batch["bin_img"]
        images = reshape_image(images)
        bin_img = reshape_image(bin_img)
        output = model(images, bboxes, bin_img)
        loss_batch = criterion(output, predi)
        loss_batch1 = loss_batch.item()
        loss = len(batch["input"]) * loss_batch1
        acc += num_true_positives(output, predi)
        num_samples += len(batch["input"])
        losses.append(loss/len(batch["input"]))
    losses = np.mean(losses)
    acc /= num_samples
    return loss, acc

if __name__ == '__main__':
    annotations = json.load(open("../Datasets/SpatialSense/Sense_new.json"))

    batch_size = 256
    nb_epochs = 10
    lr = 1e-4
    n = 18

    model = ResNet(18, True)

    model.load_state_dict(torch.load("models/*.pth"))
    model_finetune = model_vrd(model, 256, 9)
    criterion = torch.nn.CrossEntropyLoss()
    params = list(model_finetune.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)
    
    train_dataset = SpatialSense_VRD(annotations, "train")
    valid_dataset = SpatialSense_VRD(annotations, "valid")
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    train_losses = []
    valid_losses = []
    train_acc =  []
    valid_acc = []
    
    for epoch in range(nb_epochs):
        print(f"EPOCH {epoch + 1} / {nb_epochs}")
        loss, acc = finetune_model(model_finetune, train_loader, criterion, optimizer)
        train_losses.append(loss)
        print(loss, acc)

        loss, acc = valid_finetuning(model_finetune, valid_loader, criterion)
        valid_losses.append(loss)
        print(loss, acc)
    torch.save(model_finetune.state_dict(), f"models/*.pth")