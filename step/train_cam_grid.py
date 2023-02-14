
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim, nn
import cityscapesscripts.helpers.labels as labels
from ignite.metrics import ClassificationReport, Average, Precision, RunningAverage, Loss

from torchvision import transforms

import cityscapes.dataloader
from tqdm import tqdm
from net.resnet50_cam import Net
import typing

import wandb

device = torch.device("cuda")

def thresholded_output_transform(output) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y

def train(model, data_loader, optimizer: optim.Optimizer, criterion, ep, cam_num_epochs):
    model.train()
    pbar = tqdm(data_loader, desc=f'training, Epoch {ep+1}/{cam_num_epochs}')

    precision = Precision(is_multilabel=True)
    micro_precision = Precision(is_multilabel=True, average="micro")
    macro_precision = Precision(is_multilabel=True, average="macro")

    lr_metric = Average(device=torch.device("cuda"))
    loss_metric = Average(device=torch.device("cuda"))

    for img, hot_label, unique_label in pbar:
        img = img.cuda(non_blocking=True)
        unique_label = unique_label.cuda(non_blocking=True)

        pred = model(img)
        loss = criterion(pred, unique_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            loss_metric.update(loss)
            thresholded = thresholded_output_transform((pred, hot_label))
            
            precision.update(thresholded)
            micro_precision.update(thresholded)
            macro_precision.update(thresholded)
            lr_metric.update(optimizer.param_groups[0]['lr'])

            pbar.set_postfix({
                'loss' : loss.item(),
                'lr' : optimizer.param_groups[0]['lr']
            })
    
    wandb.log({
        'cam': {
            'epoch': ep,
            'train': {
                'lr' : lr_metric.compute(),
                'loss' : loss_metric.compute(),
                'precision': {labels.trainId2label[k].name: v.item() for k, v in enumerate(precision.compute()) if k < 19},
                'micro_precision': micro_precision.compute(),
                'macro_precision': macro_precision.compute()
            }
        }
    })


def validate(model, data_loader, criterion, ep):
    val_loss_meter = Average()
    
    precision = Precision(is_multilabel=True)
    macro_precision_meter = Precision(is_multilabel=True, device=torch.device("cuda"), average=True)
    micro_precision_meter = Precision(is_multilabel=True, device=torch.device("cuda"), average="micro")
    num_examples = 8
    example_images = []

    model.eval()

    with torch.no_grad():
        for (img, hot_label, unique_label) in tqdm(data_loader, desc="validating"):
            img = img.cuda(non_blocking=True)
            unique_label = unique_label.to(dtype=torch.long).cuda(non_blocking=True)

            pred = model(img)
            loss = criterion(pred, unique_label)

            thresholded = thresholded_output_transform((pred, hot_label))
            thresholded_pred, thresholded_label = thresholded

            precision.update(thresholded)
            macro_precision_meter.update(thresholded)
            micro_precision_meter.update(thresholded)
            val_loss_meter.update(loss)
        
            if len(example_images) < num_examples:
                for i in range(min(num_examples - len(example_images), img.shape[0])):
                    pred_str = ", ".join((labels.trainId2label[p.item()].name for p in thresholded_pred[i].nonzero() if p < 19))
                    label_str = ", ".join((labels.trainId2label[p.item()].name for p in thresholded_label[i].nonzero() if p < 19))
                    im = wandb.Image(img[i], caption=f"[prediction]: {pred_str}\n[label]: {label_str}")
                    example_images.append(im)

    wandb.log({
        'cam': {
            'epoch': ep,
            'val': {
                'loss' : val_loss_meter.compute(),
                'precision': {labels.trainId2label[k].name: v.item() for k, v in enumerate(precision.compute()) if k < 19},
                'macro_avg_precision': macro_precision_meter.compute(),
                'micro_avg_precision': micro_precision_meter.compute(),
                'example_image': example_images
            }
        }
    })

def run(args):
    user = "postech-cv"
    project = "irn-cityscapes"
    display_name = "train_cam_grid"

    wandb.init(
        entity=user,
        project=project,
        name=display_name,
    )
    num_workers = args.num_workers
    print(wandb.config)
    crop_size = wandb.config["cam_crop_size"]
    batch_size = wandb.config["cam_batch_size"]
    num_epoches = wandb.config["cam_num_epoches"]
    learning_rate = wandb.config["cam_learning_rate"]
    weight_decay = wandb.config["cam_weight_decay"]
    patch_size = wandb.config["patch_size"]
    cam_weights_name = args.cam_weights_name
    
    model = Net()

    train_data_loader, val_data_loader = get_dataloders(patch_size, batch_size, num_workers, crop_size)

    _, new_params = model.trainable_parameters()
    optimizer = optim.SGD(new_params, learning_rate, weight_decay=weight_decay)
    criterion = nn.MultiLabelSoftMarginLoss()

    model = torch.nn.DataParallel(model).cuda()
    model.train()
    
    for ep in range(num_epoches):
        train(model, train_data_loader, optimizer, criterion, ep, num_epoches)
        validate(model, val_data_loader, criterion, ep)
        savefile_name = cam_weights_name + f'_epoch{ep}' +'.pth'
        save_model(model, savefile_name)

    savefile_name = cam_weights_name + '.pth'
    save_model(model, savefile_name)
    torch.cuda.empty_cache()
    wandb.finish()

def save_model(model, savefile_name):
    torch.save(model.module.state_dict(), savefile_name)
    wandb.save(savefile_name)

def get_dataloders(patch_size, batch_size, num_workers, crop_size):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(crop_size + 10),
        transforms.CenterCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = cityscapes.dataloader.CityScapesDividedDataset(
        cityscapes.dataloader.Divide.Train,
        int(patch_size),
        transform=train_transform
    )
    val_dataset = cityscapes.dataloader.CityScapesDividedDataset(
        cityscapes.dataloader.Divide.Val, 
        int(patch_size), 
        transform=val_transform
    )

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    val_data_loader = DataLoader(val_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
                                 
    return train_data_loader, val_data_loader

