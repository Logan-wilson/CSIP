from models import *
from datasets import *
from training_functions import *
from SupConLoss import SupConLoss
from evaluation import * 
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    root_folder = "../Dataset/"
    sense_folder = root_folder + "/"
    dataset_folder = sense_folder + "/"
    annotation_file = sense_folder + "*.json"
    segmentation_folder = sense_folder + "/"
    fbanner_folder = sense_folder + "/"

    batch_size = 256
    nb_epochs = 10
    as_type="L"

    momentum = 0.9
    weight_decay = 0

    lr = 3e-5
    model_image = CNN_image(2, 'zeros')
    model_fbanner = CNN_fbanner(0, 'circular')
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.2)
    params = list(model_image.parameters()) + list(model_fbanner.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    annotations = json.load(open(annotation_file))[:]

    train_dataset = SpatialSense(annotations, dataset_folder, fbanner_folder, "train")
    valid_dataset = SpatialSense(annotations, dataset_folder, fbanner_folder, "valid")
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    train_losses = []
    valid_losses = []
    train_acc =  []
    valid_acc = []

    # print(len(train_dataset), len(valid_dataset))

    for epoch in range(nb_epochs):
        print(f"EPOCH {epoch + 1} / {nb_epochs}")
        #save = epoch + 1 == nb_epochs
        save = False
        loss = train_contrastive(model_image, model_fbanner, train_loader, criterion, optimizer)
        train_losses.append(loss)

        loss = valid_contrastive(model_image, model_fbanner, valid_loader, criterion)
        valid_losses.append(loss)

        # torch.save(model_image.state_dict(), f'models/model_CAE_{as_type}_{epoch}.pth')

    # torch.save(model_image.state_dict(), f'models/model_CAE_{as_type}.pth')
    plt.plot(train_losses)
    plt.plot(valid_losses)
    plt.show()

    # test
    test_dataset = SpatialSense(annotations, dataset_folder, fbanner_folder, "test")
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    embeddings, labels = test_contrastive(model_image, model_fbanner, test_loader)
    tSNE(embeddings, labels, filename='figures/tSNE.png')