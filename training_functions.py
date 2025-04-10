from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from image import reshape_image


def train_contrastive(model_img, model_fb, loader, criterion, optimizer, temperature=0.07):
    model_img.train()
    model_fb.train()
    loss = 0.0
    num_samples = 0
    for idx, batch in enumerate(tqdm(loader)):
        images = batch["input"]
        fbanners = batch["fbanner"]
        if len(images.shape) == 3:
            images = images.unsqueeze(1)
        else: 
            images = torch.permute(images, (0, 3, 1, 2))
        if len(fbanners.shape) == 3:
            fbanners = fbanners.unsqueeze(1)
        else:
            fbanners = torch.permute(fbanners, (0, 3, 1, 2))
        out_image = model_img(images)
        out_fb = model_fb(fbanners)

        labels = torch.from_numpy(np.arange(images.shape[0])).long()
        logits = (torch.mm(out_image, out_fb.T) * np.exp(temperature))
        loss_image = criterion(logits, labels)
        loss_space = criterion(torch.transpose(logits, 0, 1), labels)
        loss = (loss_image + loss_space) / 2
        num_samples += len(batch["input"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"train loss: {loss}")
    return loss.detach().numpy()


def valid_contrastive(model_img, model_fb, loader, criterion, temperature=0.07):
    model_img.eval()
    loss = 0.0
    num_samples = 0
    for idx, batch in enumerate(tqdm(loader)):
        images = batch["input"]
        if len(images.shape) == 3:
            images = images.unsqueeze(1)
        else: 
            images = torch.permute(images, (0, 3, 1, 2))
        if len(fbanners.shape) == 3:
            fbanners = fbanners.unsqueeze(1)
        else:
            fbanners = torch.permute(fbanners, (0, 3, 1, 2))
        out_image = model_img(images)
        out_fb = model_fb(fbanners)

        labels = torch.from_numpy(np.arange(images.shape[0])).long()
        logits = (torch.mm(out_image, out_fb.T) * np.exp(temperature))
        loss_image = criterion(logits, labels)
        loss_space = criterion(torch.transpose(logits, 0, 1), labels)
        loss = (loss_image + loss_space) / 2
    print(f"valid loss: {loss}")
    return loss.detach().numpy()


def test_contrastive(model_img, loader):
    model_img.eval()
    num_samples = 0
    output = []
    labels = []
    for idx, batch in enumerate(tqdm(loader)):
        images = batch["input"]
        relations = batch["rel"]
        if len(images.shape) == 3:
            images = images.unsqueeze(1)
        else: 
            images = torch.permute(images, (0, 3, 1, 2))
        if len(fbanners.shape) == 3:
            fbanners = fbanners.unsqueeze(1)
        else:
            fbanners = torch.permute(fbanners, (0, 3, 1, 2))
        out_image = model_img(images)
        # out_fb = model_fb(fbanners)
        for i in range(out_image.shape[0]):
            output.append(out_image[i].detach().numpy())
            labels.append(relations[i])
    return output, labels


def finetuning_training(model_img, loader, criterion, optimizer, temperature=0.07, freeze_layers=False):
    model_img.train()
    for param in model_img.parameters():
        param.requires_grad = not freeze_layers
    model_img.fc.weight.requires_grad = True
    model_img.fc.bias.requires_grad = True

    loss = 0.0
    num_samples = 0
    for idx, batch in enumerate(tqdm(loader)):
        images = batch["input"]
        fbanners = batch["fbanner"]
        if len(images.shape) == 3:
            images = images.unsqueeze(1)
        else: 
            images = torch.permute(images, (0, 3, 1, 2))
        out_image = model_img(images)
        labels = torch.from_numpy(np.arange(images.shape[0])).long()
        logits = (torch.mm(out_image, fbanners.T) * np.exp(temperature))
        loss_image = criterion(logits, labels)
        loss_space = criterion(torch.transpose(logits, 0, 1), labels)
        loss = (loss_image + loss_space) / 2
        num_samples += len(batch["input"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"train loss: {loss}")
    return loss.detach().numpy()




def train_cont_spatial(model_img, model_fb, loader, criterion, optimizer, temperature=0.07, freeze_layers=False):
    model_img.train()
    model_fb.train()

    losses = []
    loss = 0.0
    num_samples = 0
    for idx, batch in enumerate(tqdm(loader)):
        images = batch["input"]
        bboxes = batch["bboxes"]
        bin_img = batch["bin_img"]
        fbanners = batch["fbanner"]
        # print(bin_img.shape)
        bin_img = reshape_image(bin_img)
        images = reshape_image(images)
        fbanners = reshape_image(fbanners)
        # out_image = model_img(images, bboxes)
        _, out_image = model_img(images, bin_img)
        out_fb = model_fb(fbanners)

        labels = torch.from_numpy(np.arange(images.shape[0])).long()
        logits = (torch.mm(out_image, out_fb.T) * np.exp(temperature))
        loss_image = criterion(logits, labels)
        loss_space = criterion(torch.transpose(logits, 0, 1), labels)
        loss = (loss_image + loss_space) / 2
        # loss = loss_image
        num_samples += len(batch["input"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().numpy())
    print(f"train loss: {np.mean(losses)}")
    return np.mean(losses)


def valid_cont_spatial(model_img, model_fb, loader, criterion, temperature=0.07):
    model_img.eval()
    model_fb.eval()
    losses = []
    loss = 0.0
    num_samples = 0
    for idx, batch in enumerate(tqdm(loader)):
        images = batch["input"]
        bboxes = batch["bboxes"]
        bin_img = batch["bin_img"]
        fbanners = batch["fbanner"]
        # print(bin_img.shape)
        bin_img = reshape_image(bin_img)
        images = reshape_image(images)
        fbanners = reshape_image(fbanners)
        # out_image = model_img(images, bboxes)
        _, out_image = model_img(images, bin_img)
        out_fb = model_fb(fbanners)

        labels = torch.from_numpy(np.arange(images.shape[0])).long()
        logits = (torch.mm(out_image, out_fb.T) * np.exp(temperature))#.unsqueeze(1)
        loss_image = criterion(logits, labels)
        loss_space = criterion(torch.transpose(logits, 0, 1), labels)
        loss = (loss_image + loss_space) / 2
        losses.append(loss.detach().numpy())
    mean_loss = np.mean(losses)
    print(f"valid loss: {mean_loss}")
    return mean_loss