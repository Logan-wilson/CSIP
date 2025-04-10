from models import *
from evaluation import *
from datasets import *
from baselines import * 
from baseline_COCO import model2D_BL
import torch
import json

symmetry = True

root = "../Datasets"
senseroot = f"{root}/"
unrelroot = f"{root}/"
sense_symmetry = "SFB" if symmetry else "fbanner"
unrel_symmetry = "_sym" if symmetry else ""


def CBIR_outputs(model, loader, model_name):
    model.eval()
    num_samples = 0
    outputs = []
    obj1 = []
    obj2 = []
    labels = [] 
    spatial = []
    for idx, batch in enumerate(tqdm(loader)):
        images = batch["input"]
        fbanners = batch["fbanner"]
        relations = batch["rel"]
        subj = batch["subj"]
        obj = batch["obj"]
        bboxes = batch["bboxes"]
        if model_name == "resnet":
            images = reshape_image(images)
            out = model(images)
        if model_name == "resnet_bl":
            images = reshape_image(images)
            out, lab = model(images)
        elif model_name == "2D":
            bboxes = batch["bboxes"]
            out = model(bboxes)
            out = out[0]
        elif model_name == "cont":
            images = reshape_image(images)
            out = model(images, bboxes)
        elif model_name == "drnet":
            images = reshape_image(images)
            masks = batch["bin_img"]
            masks = masks.unsqueeze(1)
            out, _ = model(images, masks)
        elif model_name == "vit":
            images = reshape_image(images)
            out = model(images)
        for i in range(out.shape[0]):
            outputs.append(out[i].detach().numpy())
            spatial.append(fbanners[i].detach().numpy().flatten())
            labels.append(relations[i])
            obj1.append(subj[i])
            obj2.append(obj[i])
            num_samples += 1
    return outputs, labels, obj1, obj2, spatial


def qualitative(model, loader, k=5, modelname=""):
    model.eval()
    outputs = []
    names = []
    for idx, batch in enumerate(tqdm(loader)):
        images = batch["input"]
        name = batch["imagename"]
        bboxes = batch["bboxes"]
        if modelname == "resnet":
            images = reshape_image(images)
            out = model(images)
        if modelname == "resnet_bl":
            images = reshape_image(images)
            out, lab = model(images)
        elif modelname == "2D":
            bboxes = batch["bboxes"]
            out = model(bboxes)
            out = out[0]
        elif modelname == "cont":
            images = reshape_image(images)
            out = model(images, bboxes)
        elif modelname == "drnet":
            images = reshape_image(images)
            masks = batch["bin_img"]
            masks = masks.unsqueeze(1)
            out, _ = model(images, masks)
        for i in range(out.shape[0]):
            outputs.append(out[i].detach().numpy())
            names.append(name[i])
    to_save = []
    matrix = cosine_similarity(outputs, Y=None)
    for i in range(len(outputs)):
        indices = np.argsort(matrix[i])
        ranked = []
        for j in range(k):
            ranked.append(names[indices[-(j+1)]])
        to_save.append(ranked)
    print(to_save[0][0], names[0])
    save_pickle(f"*.pkl", to_save)
    return outputs


def ndcg(ns, func, outputs, additionnal_data, rounder=3):
    n0 = func(ns[0], outputs, additionnal_data)
    n1 = func(ns[1], outputs, additionnal_data)
    n2 = func(ns[2], outputs, additionnal_data)
    n3 = func(ns[3], outputs, additionnal_data)
    n4 = func(ns[4], outputs, additionnal_data)
    print(round(n0, rounder), round(n1, rounder), round(n2, rounder), round(n3, rounder), round(n4, rounder))
    f, ax = plt.subplots(1)
    ax.plot([1, 3, 5, 10, 25], [n0, n1, n2, n3, n4])
    ax.set_ylim([0, 1])
    plt.show()


if __name__ == '__main__':
    modelname = ""
    modelsource = ""
    fbanners = load_pickle("models/*.pkl")
    if modelname == "resnet":
        model = ResNet(34, True)
        model.load_state_dict(torch.load("models/*.pth"))
    if modelname == "resnet_bl":
        model = CNN_spatial(5)
        model.load_state_dict(torch.load("models/*.pth"))
    if modelname == "cont":
        model = cont_spatial()
        model.load_state_dict(torch.load("models/*.pth"))
    if modelname == "vtranse":
        pass
    if modelname == "drnet":
        model = DeepRelationalNetwork(256)
        model.load_state_dict(torch.load("models/*.pth"))
    if modelname == "2D":
        model = model2D_BL(8, 256, 256)
        modelsource = "*.pth"
    if modelname == "vit":
        model = ViT()
    if modelname == "cont":
        model = cont_spatial(final_layer=5)
        model.load_state_dict(torch.load("models/*.pth"))
    if modelsource != "": model.load_state_dict(torch.load(f"models/{modelsource}"))
    sense_annotations = json.load(open(f"{senseroot}/*.json"))[:]
    unrel_annotations = json.load(open(f"{unrelroot}/*.json"))[:]
    print(len(sense_annotations))
    dataset = Unrel(unrel_annotations, f"{unrelroot}/", f"/")
    
    print(len(dataset))
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, shuffle=False, num_workers=4)

    evaluation_type = "quantitative"  

    if evaluation_type == "quantitative":
        outputs, relations, subj, obj, spatial = CBIR_outputs(model, dataset_loader, modelname)
        spatial_data = spatial_classes(spatial, 0)
        ndcg([1,3,5,10,25], NDCG_spatial, outputs, spatial_data)

    if evaluation_type == "qualitative":
        outputs = qualitative(model, dataset_loader, 5, setname="sense", modelname=modelname)
