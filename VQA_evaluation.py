from datasets import GQA_rel
from evaluation import * 
import json
import torch
from tqdm import tqdm
import numpy as np
from transformers import BertModel, BertTokenizer
from image import reshape_image


def ResNet(n_layers):
    model = torch.hub.load('pytorch/vision:v0.10.0', f'resnet{n_layers}', pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    return feature_extractor

def ResNet18():
    model = torch.hub.load('pytorch/vision:v0.10.0', f'resnet18', pretrained=True)
    model.fc = torch.nn.Linear(in_features=512, out_features=256, bias=True)
    return model

def BERT(dtype):
    model = BertModel.from_pretrained("bert-base-uncased", torch_dtype=dtype, attn_implementation="sdpa")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer


class model_vqa(torch.nn.Module):
    def __init__(self, image_features, text_features):
        super(model_vqa, self).__init__()
        self.linear1 = torch.nn.Linear(image_features.shape[1], 256)
        self.linear2 = torch.nn.Linear(256, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 2)

        self.batchnorm1 = torch.nn.BatchNorm1d(256)
        self.batchnorm2 = torch.nn.BatchNorm1d(256)
        self.batchnorm3 = torch.nn.BatchNorm1d(128)
        self.activation = torch.nn.ReLU()


    def forward(self, features):
        x1 = self.batchnorm1(self.activation(self.linear1(features)))
        x2 = self.batchnorm2(self.activation(self.linear2(x1)))
        x3 = self.batchnorm3(self.activation(self.linear3(x2)))
        x4 = self.linear4(x3)
        return x4

def bert_embeddings(bert_model, tokenizer, questions):
    tokens = tokenizer(questions, padding=True, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(**tokens)
        word_embedding = outputs.last_hidden_state[:,0]
    return word_embedding


def train_vqa(model, image_encoder, spatial_encoder, text_encoder, tokenizer, loader, criterion, optimizer):
    model.train()
    losses = []
    acc = 0.0
    num_samples = 0
    for idx, batch in enumerate(tqdm(loader)):
        images = batch['image']
        questions = batch['question']
        answers = batch['answer']
        images = reshape_image(images)
        with torch.no_grad():
            text_features = bert_embeddings(text_encoder, tokenizer, questions)
            vision_features = torch.squeeze(image_encoder(images))
            spatial_features = torch.squeeze(spatial_encoder(images))
        # features = torch.cat((vision_features, text_features, spatial_features), 1)
        features = vision_features * text_features * spatial_features
        output = model(features)
        loss_batch = criterion(output, answers)
        loss_batch1 = loss_batch.item()
        loss = len(answers) * loss_batch1
        acc += num_true_positives(output, answers)
        num_samples += len(answers)
        losses.append(loss/len(answers))
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
    losses = np.mean(losses)
    acc /= num_samples
    return losses, acc


def valid_vqa(model, image_encoder, spatial_encoder, text_encoder, tokenizer, loader, criterion):
    model.eval()
    losses = []
    acc = 0.0
    num_samples = 0
    for idx, batch in enumerate(tqdm(loader)):
        images = batch['image']
        questions = batch['question']
        answers = batch['answer']
        images = reshape_image(images)
        with torch.no_grad():
            text_features = bert_embeddings(text_encoder, tokenizer, questions)
            vision_features = torch.squeeze(image_encoder(images))
            spatial_features = torch.squeeze(spatial_encoder(images))
            # features = torch.cat((vision_features, text_features, spatial_features), 1)
            features = vision_features * text_features * spatial_features
            output = model(features)
        loss_batch = criterion(output, answers)
        loss_batch1 = loss_batch.item()
        loss = len(answers) * loss_batch1
        acc += num_true_positives(output, answers)
        num_samples += len(answers)
        losses.append(loss/len(answers))
    losses = np.mean(losses)
    acc /= num_samples
    return losses, acc


def test_vqa(model, image_encoder, spatial_encoder, text_encoder, tokenizer, loader):
    model.eval()
    acc = 0.0
    num_samples = 0
    for idx, batch in enumerate(tqdm(loader)):
        images = batch['image']
        questions = batch['question']
        answers = batch['answer']
        images = reshape_image(images)
        with torch.no_grad():
            text_features = bert_embeddings(text_encoder, tokenizer, questions)
            vision_features = torch.squeeze(image_encoder(images))
            spatial_features = torch.squeeze(spatial_encoder(images))
            # features = torch.cat((vision_features, text_features, spatial_features), 1)
            features = vision_features * text_features * spatial_features
            output = model(features)
        acc += num_true_positives(output, answers)
        num_samples += len(answers)
    acc /= num_samples
    return acc

if __name__ == '__main__':
    ant_path = '../Dataset/GQA/*.json'
    img_path = '../Dataset/GQA/images_folder'
    annotations = json.load(open(ant_path))
    keys = list(annotations.keys())
    print(len(keys))
    train_keys = keys[:int(len(keys) * 0.80)]
    valid_keys = keys[int(len(keys) * 0.80):]
    print(len(train_keys), len(valid_keys))

    batch_size = 64
    lr = 3e-5

    train_set = GQA_rel(annotations, train_keys, img_path)
    valid_set = GQA_rel(annotations, valid_keys, img_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False, num_workers=0)


    model = model_vqa(768, 512+256)
    params = list(model.parameters())
    model_image = ResNet(18) # pretrained/supervised
    model_spatial = ResNet18() #csip version
    model_spatial.load_state_dict(torch.load("models/CSIP.pth"))
    model_text, tokenizer = BERT(torch.float32)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params, lr=lr)

    n_epochs = 3
    for epoch in range(n_epochs):
        print()
        print(f'epoch {epoch}')
        loss, acc = train_vqa(model, model_image, model_spatial, model_text, tokenizer, train_loader, criterion, optimizer)
        print(f"train epoch {epoch+1}: loss/acc ", loss, acc)
        loss, acc = valid_vqa(model, model_image, model_spatial, model_text, tokenizer, valid_loader, criterion)
        print(f"valid epoch {epoch+1}: loss/acc ", loss, acc)

    torch.save(model.state_dict(), f"models/*.pth")

    ant_path_test = '../Dataset/GQA/*.json'
    test_annotations = json.load(open(ant_path_test))
    test_keys = list(test_annotations.keys())
    test_set = GQA_rel(test_annotations, test_keys, img_path)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    acc = test_vqa(model, model_image, model_spatial, model_text, tokenizer, test_loader)
    print(acc)