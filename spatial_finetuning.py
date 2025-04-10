from models import *
from datasets import *
from training_functions import *
from evaluation import *
import torch
from tqdm import tqdm
import numpy as np
import json

if __name__ == "__main__":
    root_folder = "../Datasets/"
    sense_folder = root_folder + "SpatialSense/"
    dataset_folder = sense_folder + "images/flickr/"
    annotation_file = sense_folder + "sense_new.json"
    segmentation_folder = sense_folder + "segmentations/cropped"
    fbanner_folder = sense_folder + "segmentations/SFB"

    fbanners = load_pickle("models/fbanners_PCA_256.pkl")

    batch_size = 256
    nb_epochs = 20
    as_type="L"

    momentum = 0.9
    weight_decay = 0

    lr = 1e-3
    n = 18
    model_image = ResNet(n, False)
    # model_fbanner = CNN_fbanner(0, 'circular')
    # criterion = SupConLoss(temperature=0.5, base_temperature=0.5)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.15)
    params = list(model_image.parameters())# + list(model_fbanner.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)

    annotations = json.load(open(annotation_file))[:]

    train_dataset = SpatialSense(annotations, "train", fbanners)
    valid_dataset = SpatialSense(annotations, "valid", fbanners)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    train_losses = []
    valid_losses = []
    train_acc =  []
    valid_acc = []

    for epoch in range(nb_epochs):
        print(f"EPOCH {epoch + 1} / {nb_epochs}")
        #save = epoch + 1 == nb_epochs
        save = False
        loss = finetuning_training(model_image, None, train_loader, criterion, optimizer, freeze_layers=False)
        train_losses.append(loss)

        loss = valid_contrastive(model_image, None, valid_loader, criterion)
        valid_losses.append(loss)

        # torch.save(model_image.state_dict(), f'models/model_CAE_{as_type}_{epoch}.pth')

    torch.save(model_image.state_dict(), f'models/ResNet{n}_finetuned.pth')
    f, ax = plt.subplots(1)
    ax.plot(train_losses)
    ax.plot(valid_losses)
    ax.set_ylim(bottom=0)
    plt.show()

    # test
    test_dataset = SpatialSense(annotations, "test", fbanners)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    embeddings, labels = test_contrastive(model_image, test_loader)
    tSNE(embeddings, labels, filename='figures/tSNE.png')