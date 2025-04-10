import torch
from tqdm import tqdm
from models import *
import numpy as np
from image import *
import json
from evaluation import *
from SRR_finetuning import SpatialSense_VRD


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


    def forward(self, image, bboxes):
        embedding = self.model(image, bboxes)
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
                bin_img_0 = load_fbanner("../Dataset/path", imagename + "_0", 224, "L", "png").astype(np.float32, copy=False)
                bin_img_1 = load_fbanner("../Dataset/path", imagename + "_1", 224, "L", "png").astype(np.float32, copy=False)
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
                        "bboxes": np.array(bboxes).astype(np.float32, copy=False)
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


def test_vrd(model0, loader):
    model0.eval()
    acc = 0.0
    num_samples = 0
    outputs = []
    relations = []
    predictions = []
    acc1 = 0
    acc2 = 0
    acc3 = 0
    acc4 = 0
    acc5 = 0

    for idx, batch in enumerate(tqdm(loader)):
        images = batch["input"]
        bboxes = batch["bboxes"]
        bin_img = batch["bin_img"]
        predi = batch["pred_rel"]
        rel = batch["rel"]
        images = reshape_image(images)
        bin_img = reshape_image(bin_img)
        with torch.no_grad():
            output = model0(images, bboxes)
        acc += num_true_positives(output, predi)
        a1, a2, a3, a4, a5 = acc_at_k(output, predi, 5)
        acc1 += a1
        acc2 += a2
        acc3 += a3
        acc4 += a4
        acc5 += a5
        num_samples += len(batch["input"])
        for i in range(output.shape[0]):
            outputs.append(torch.Tensor.numpy(output[i]))
            relations.append(rel[i])
            predictions.append(torch.argmax(output[i]))

    acc /= num_samples
    acc1 /= num_samples
    acc2 /= num_samples
    acc3 /= num_samples
    acc4 /= num_samples
    acc5 /= num_samples
    return acc, outputs, relations, predictions, [acc1, acc1 + acc2, acc1 + acc2 + acc3, acc1 + acc2 + acc3 + acc4, acc1 + acc2 + acc3+acc4+acc5]

if __name__ == '__main__':
    annotations = json.load(open("../Datasets/SpatialSense/Sense_new.json"))

    batch_size = 256
    n = 18
    model = cont_spatial()
    model_finetune = model_vrd(model, 256, 9)
    model_finetune.load_state_dict(torch.load("models/finetuning/*.pth"))
# 
    test_dataset = SpatialSense_VRD(annotations, "test")
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    acc, embeddings, labels, predis, cumul_accs = test_vrd(model_finetune, test_loader)
    predicates = ["above", "behind", "in", "in front of", "next to", "on", "to the left of", "to the right of", "under"]
    predicts = []
    for i in range(len(predis)):
        predicts.append(predicates[predis[i]])
    print(acc)
    print(cumul_accs)
    tSNE(embeddings, labels, filename=f"figures/tsne.png")
    torch.save(model.state_dict(), f"modelpath/*.pth")