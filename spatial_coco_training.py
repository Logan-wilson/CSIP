from models import *
from datasets import *
from training_functions import *
from evaluation import *
import torch
from tqdm import tqdm
import numpy as np
import json

if __name__ == "__main__":
    root_folder = "../Dataset/COCO"
    dataset_folder = root_folder + "/train2017"
    annotation_file = root_folder + "/"
    fbanner_folder = root_folder + "/"

    batch_size = 256
    nb_epochs = 10

    lr = 2e-3
    n = 18
    model_image = ResNet(n, False)
    model_fbanner = CNN_fbanner(0, 'circular')
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.3)
    params = list(model_image.parameters()) + list(model_fbanner.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)

    annotations = json.load(open(annotation_file))[:]
    indexes = list(range(len(annotations)))
    random.shuffle(indexes)
    split = [0.80, 0.20, 0.0]
    print(len(indexes))
    train_indexes = indexes[:int(split[0]*len(indexes))]
    valid_indexes = indexes[int(split[0]*len(indexes)):int((split[0] + split[1])*len(indexes))]
    test_indexes =  indexes[int((split[0] + split[1])*len(indexes)):]
    print(f"Effective train split = {len(train_indexes)/len(indexes)*100}%")
    print(f"Effective val split = {len(valid_indexes)/len(indexes)*100}%")

    train_dataset = SenseCOCO("train")
    valid_dataset = SenseCOCO("valid")
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    train_losses = []
    valid_losses = []
    train_acc =  []
    valid_acc = []

    for epoch in range(nb_epochs):
        print(f"EPOCH {epoch + 1} / {nb_epochs}")
        loss = train_cont_spatial(model_image, model_fbanner, train_loader, criterion, optimizer, freeze_layers=False)
        train_losses.append(loss)

        loss = valid_cont_spatial(model_image, model_fbanner, valid_loader, criterion)
        valid_losses.append(loss)

    torch.save(model_image.state_dict(), f"models/*.pth")
    f, ax = plt.subplots(1)
    ax.plot(train_losses)
    ax.plot(valid_losses)
    ax.set_ylim(bottom=0)
    plt.show()