import torch
import torchvision


class CNN_fbanner(torch.nn.Module):

    def __init__(self, padding=0, paddingmode="zeros", bias=True):
        super(CNN_fbanner, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 28, 7, padding=padding, padding_mode=paddingmode, bias=bias)
        self.conv2 = torch.nn.Conv2d(16, 16, 5, 2, padding=padding, padding_mode=paddingmode, bias=bias)
        self.linear1 = torch.nn.Linear(16*13*13, 1024)
        self.linear2 = torch.nn.Linear(1024, 256)

        self.batchnorm1 = torch.nn.BatchNorm2d(16)
        self.batchnorm2 = torch.nn.BatchNorm2d(16)

        self.activation = torch.nn.ReLU()

    def forward(self, image):
        x1 = self.batchnorm1(self.activation(self.conv1(image)))
        x2 = self.batchnorm2(self.activation(self.conv2(x1)))
        x2 = torch.flatten(x2, start_dim=1)
        x3 = self.activation(self.linear1(x2))
        x4 = self.activation(self.linear2(x3))
        return x4


class CNN_image(torch.nn.Module):

    def __init__(self, padding=0, paddingmode="zeros", bias=True):
        super(CNN_image, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 28, 5, padding=padding, padding_mode=paddingmode, bias=bias)
        self.conv2 = torch.nn.Conv2d(6, 8, 13, 3, padding=padding, padding_mode=paddingmode, bias=bias)
        self.conv3 = torch.nn.Conv2d(8, 12, 11, 1, padding=padding, padding_mode=paddingmode, bias=bias)
        self.conv4 = torch.nn.Conv2d(12, 15, 7, 1, padding=padding, padding_mode=paddingmode, bias=bias)
        self.conv5 = torch.nn.Conv2d(15, 18, 3, 1, padding=padding, padding_mode=paddingmode, bias=bias)
        self.conv6 = torch.nn.Conv2d(18, 18, 3, 1, padding=padding, padding_mode=paddingmode, bias=bias)
        self.linear = torch.nn.Linear(18*5*5, 256)

        self.pool = torch.nn.AvgPool2d(3)
        self.batchnorm1 = torch.nn.BatchNorm2d(6)
        self.batchnorm2 = torch.nn.BatchNorm2d(8)
        self.batchnorm3 = torch.nn.BatchNorm2d(12)
        self.batchnorm4 = torch.nn.BatchNorm2d(15)
        self.batchnorm5 = torch.nn.BatchNorm2d(18)
        self.batchnorm6 = torch.nn.BatchNorm2d(18)

        self.activation = torch.nn.ReLU()

    def forward(self, image):
        x1 = self.batchnorm1(self.activation(self.conv1(image)))
        x2 = self.batchnorm2(self.activation(self.conv2(x1)))
        x3 = self.batchnorm3(self.activation(self.conv3(x2)))
        x4 = self.batchnorm4(self.activation(self.conv4(x3)))
        x5 = self.batchnorm5(self.activation(self.conv5(x4)))
        
        x6 = torch.flatten(x5, start_dim=1)
        x7 = self.linear(x6)
        return x7
    

def ResNet(n_layers, pretrained, out_features=256):
    model = torch.hub.load('pytorch/vision:v0.10.0', f'resnet{n_layers}', pretrained=pretrained)
    if out_features != 0:
        model.fc = torch.nn.Linear(in_features=512, out_features=out_features, bias=True)
    return model



class cont_spatial(torch.nn.Module):

    def __init__(self, final_layer=256):
        super(cont_spatial, self).__init__()
        self.resnet = ResNet(18, True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc.weight.requires_grad = True
        self.resnet.fc.bias.requires_grad = True
        
        self.bbox_layer1 = torch.nn.Linear(8, 32)
        self.batchnorm1 = torch.nn.BatchNorm1d(32)
        self.bbox_layer2 = torch.nn.Linear(32, 64)
        self.batchnorm2 = torch.nn.BatchNorm1d(64)

        self.fc1 = torch.nn.Linear(320, 512)
        self.fc_batchnorm = torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(512, final_layer)

        self.relu = torch.nn.ReLU()
    
    def forward(self, image, bboxes):

        out_image = self.resnet(image)
        x1 = self.batchnorm1(self.relu(self.bbox_layer1(bboxes)))
        out_bbox = self.batchnorm2(self.relu(self.bbox_layer2(x1)))
        cat_outs = torch.cat((out_image, out_bbox), 1)
        y1 = self.fc_batchnorm(self.relu(self.fc1(cat_outs)))
        y2 = self.relu(self.fc2(y1))
        return y2
    

def ViT():
    vit_weights = torchvision.models.vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
    vit_model = torchvision.models.vision_transformer.vit_b_16(vit_weights)
    return vit_model


class DeepRelationalNetwork(torch.nn.Module):

    def __init__(self, n_labels):
        super().__init__()
        self.visual_module = ResNet(18, True)
        for param in self.visual_module.parameters():
            param.requires_grad = False
        self.visual_module.fc.requires_grad = True
        self.fc1 = torch.nn.Linear(256, 256)
        self.conv1 = torch.nn.Conv2d(2, 4, 13, 5)
        self.batchnorm1 = torch.nn.BatchNorm2d(4)
        self.conv2 = torch.nn.Conv2d(4, 8, 9, 3)
        self.batchnorm2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 8, 9)
        self.batchnorm3 = torch.nn.BatchNorm2d(8)
        self.relu = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(256 + 128, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc_final = torch.nn.Linear(256, n_labels)


    def forward(self, image, bin_img):
        out_image = self.visual_module(image)

        x_s1 = self.batchnorm1(self.relu(self.conv1(bin_img)))
        x_s2 = self.batchnorm2(self.relu(self.conv2(x_s1)))
        x_s3 = self.batchnorm3(self.relu(self.conv3(x_s2)))
        out_spatial = torch.flatten(x_s3, start_dim=1)
        conc_tensor = torch.cat((out_image, out_spatial), 1)
        f1 = self.relu(self.fc1(conc_tensor))
        f2 = self.relu(self.fc2(f1))
        return f2, self.relu(self.fc_final(f2))
    