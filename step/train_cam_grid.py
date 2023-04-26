
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader, random_split
from torch import optim, nn
import cityscapesscripts.helpers.labels as labels
from torchmetrics import MeanMetric

from torchvision import transforms

import cityscapes.divided_dataset
from tqdm import tqdm
from net.resnet50_cam_lightning import Net
import typing

from voc12.dataloader import CAT_LIST

import wandb

import pytorch_lightning as pl
from torchmetrics import Precision

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

device = torch.device("cuda")

# def thresholded_output_transform(output) -> typing.Tuple[torch.Tensor, torch.Tensor]:
#     y_pred, y = output
#     result = torch.zeros_like(y_pred, dtype=torch.int32)
#     result[y_pred >= 0.2] = 1
#     return result, y

# def train(model, data_loader, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, criterion, ep, cam_num_epochs):
#     model.train()
#     pbar = tqdm(data_loader, desc=f'training, Epoch {ep+1}/{cam_num_epochs}')

#     precision = Precision('multilabel', threshold=a)
#     micro_precision = Precision('multilabel', average="micro")
#     macro_precision = Precision('multilabel', average="macro")
#     running_precision = Precision('multilabel')
#     running_micro_precision = Precision('multilabel', average="micro")
#     running_macro_precision = Precision('multilabel', average="macro")

#     lr_metric = MeanMetric(device=torch.device("cuda"))
#     loss_metric = MeanMetric(device=torch.device("cuda"))

#     for i, (img, hot_label, unique_label, _, _, _, _) in enumerate(pbar):
#         img = img.cuda(non_blocking=True)
#         hot_label = hot_label.cuda(non_blocking=True)
#         unique_label = unique_label.cuda(non_blocking=True)

#         weights = torch.full((20, ), 1.0, device=device)
#         for i in (0, 1, 2, 8):
#             weights[i] = 0.05

#         pred = model(img)
#         loss = criterion(pred[:, :19], hot_label[:, :19], weights[:19])

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if bool(loss.isnan()) or bool(loss.isinf()):
#             return loss

#         with torch.no_grad():
#             loss_metric.update(loss)
#             thresholded = thresholded_output_transform((pred, hot_label))
            
#             precision.update(thresholded)
#             micro_precision.update(thresholded)
#             macro_precision.update(thresholded)
#             running_precision.update(thresholded)
#             running_micro_precision.update(thresholded)
#             running_macro_precision.update(thresholded)
#             lr_metric.update(optimizer.param_groups[0]['lr'])

#             pbar.set_postfix({
#                 'loss' : loss.item(),
#                 'lr' : optimizer.param_groups[0]['lr']
#             })
#             wandb.log({
#                 'cam.running': {
#                     'epoch': ep,
#                     'train': {
#                         'lr' : optimizer.param_groups[0]['lr'],
#                         'loss' : loss.item(),
#                         'precision': precision_to_cityscape_dict(torch.any(hot_label, dim=0), running_precision.compute()),
#                         'micro_precision': running_micro_precision.compute(),
#                         'macro_precision': running_macro_precision.compute()
#                     }
#                 }
#             })
#         running_precision.reset()
#         running_micro_precision.reset()
#         running_macro_precision.reset()
    
#     wandb.log({
#         'cam': {
#             'epoch': ep,
#             'train': {
#                 'lr' : lr_metric.compute(),
#                 'loss' : loss_metric.compute(),
#                 'precision': precision_to_cityscape_dict(torch.ones((20, )), precision.compute()),
#                 'micro_precision': micro_precision.compute(),
#                 'macro_precision': macro_precision.compute()
#             }
#         }
#     })

#     return loss_metric.compute()

# def hot_to_cat_str(hot):
#     return ", ".join((labels.trainId2label[p.item()].name for p in hot.nonzero()))

# def precision_to_cat_dict(exist_class, precision):
#     return {CAT_LIST[k] : v.item() for k, v in enumerate(precision) if bool(exist_class[k])}

# def hot_to_cityscape_str(hot):
#     return ", ".join((labels.trainId2label[p.item()].name for p in hot.nonzero() if p < 19))

# def precision_to_cityscape_dict(exist_class, precision):
#     return {labels.trainId2label[k].name : v.item() for k, v in enumerate(precision) if k < 19 and bool(exist_class[k])}


# def validate(model, data_loader, criterion, ep):
#     val_loss_meter = MeanMetric()
    
#     precision = Precision(is_multilabel=True)
#     macro_precision_meter = Precision(is_multilabel=True, device=torch.device("cuda"), average=True)
#     micro_precision_meter = Precision(is_multilabel=True, device=torch.device("cuda"), average="micro")
#     num_examples = 8
#     example_images = []

#     model.eval()

#     with torch.no_grad():
#         for (img, hot_label, unique_label) in tqdm(data_loader, desc="validating"):
#             img = img.cuda(non_blocking=True)
#             unique_label = unique_label.to(dtype=torch.long).cuda(non_blocking=True)
#             pred = model(img)
#             loss = criterion(pred, unique_label)

#             thresholded = thresholded_output_transform((pred, hot_label))
#             thresholded_pred, thresholded_label = thresholded

#             precision.update(thresholded)
#             macro_precision_meter.update(thresholded)
#             micro_precision_meter.update(thresholded)
#             val_loss_meter.update(loss)
        
#             if len(example_images) < num_examples:
#                 for i in range(min(num_examples - len(example_images), img.shape[0])):
#                     pred_str = hot_to_cityscape_str(thresholded_pred[i])
#                     label_str = hot_to_cityscape_str(hot_label[i])
#                     im = wandb.Image(img[i], caption=f"[prediction]: {pred_str}\n[label]: {label_str}")
#                     example_images.append(im)

#     wandb.log({
#         'cam': {
#             'epoch': ep,
#             'val': {
#                 'loss' : val_loss_meter.compute(),
#                 'precision': precision_to_cityscape_dict(torch.ones((20, )), precision.compute()),
#                 'macro_avg_precision': macro_precision_meter.compute(),
#                 'micro_avg_precision': micro_precision_meter.compute(),
#                 'example_image': example_images
#             }
#         }
#     })

def run(args):
    user = "postech-cv"
    project = "irn-cityscapes"
    display_name = "train_cam_grid_no_sigmoid"

    wandb.init(
        entity=user,
        project=project,
        name=display_name,
    )
    num_workers = args.num_workers
    print(wandb.config)
    crop_size = args.cam_crop_size
    batch_size = args.cam_batch_size
    num_epoches = args.cam_num_epoches
    learning_rate = args.cam_learning_rate
    weight_decay = args.cam_weight_decay
    patch_size = args.patch_size
    cam_weights_name = args.cam_weights_name
    
    model = Net(learning_rate, weight_decay, 0.5)
    datamodule = get_cityscapes_datamodule(patch_size, batch_size, crop_size, num_workers)

    logger = WandbLogger(project="irn-cityscapes", name="train_cam_grid_no_sigmoid")
    checkpoint_callback = ModelCheckpoint(dirpath='models/train', filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}' )
    
    trainer = pl.Trainer(logger=logger, max_epochs=num_epoches, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule)
    
    savefile_name = cam_weights_name + '.pth'
    save_model(model, savefile_name)
    torch.cuda.empty_cache()
    wandb.finish()

# def run_legacy():
#     user = "postech-cv"
#     project = "irn-cityscapes"
#     display_name = "train_cam_grid_no_sigmoid"

#     wandb.init(
#         entity=user,
#         project=project,
#         name=display_name,
#     )
#     num_workers = args.num_workers
#     print(wandb.config)
#     crop_size = args.cam_crop_size
#     batch_size = args.cam_batch_size
#     num_epoches = args.cam_num_epoches
#     learning_rate = args.cam_learning_rate
#     weight_decay = args.cam_weight_decay
#     patch_size = args.patch_size
#     cam_weights_name = args.cam_weights_name
    
#     model = Net()

#     datamodule = get_cityscapes_datamodule(patch_size, batch_size, num_workers, crop_size)

#     _, new_params = model.trainable_parameters()
#     optimizer = optim.SGD(new_params, learning_rate, weight_decay=weight_decay, momentum=0.9)
#     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, step_size_up=2, step_size_down=3, mode='triangular')
#     criterion = torch.nn.functional.multilabel_soft_margin_loss

#     logger = WandbLogger(project="irn-cityscapes", name="train_cam_grid_no_sigmoid")
#     trainer = pl.Trainer(logger=logger, max_epochs=num_epoches)

#     trainer.fit(model, datamodule)

#     model = torch.nn.DataParallel(model).cuda()
#     model.train()
    
#     for ep in range(num_epoches):
#         loss = train(model, train_data_loader, optimizer, scheduler, criterion, ep, num_epoches)
#         if bool(loss.isnan()) or bool(loss.isinf()):
#             return
#         validate(model, val_data_loader, criterion, ep)
#         savefile_name = cam_weights_name + f'_epoch{ep}' +'.pth'
#         save_model(model, savefile_name)
#         scheduler.step()

#     savefile_name = cam_weights_name + '.pth'
#     save_model(model, savefile_name)
#     torch.cuda.empty_cache()
#     wandb.finish()

def save_model(model, savefile_name):
    torch.save(model.module.state_dict(), savefile_name)
    wandb.save(savefile_name)

def get_cityscapes_datamodule(patch_size, batch_size, crop_size, num_workers):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(crop_size, antialias=False), # 넣었다 뺐다
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(crop_size + 10, antialias=False), # type: ignore
        transforms.CenterCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = cityscapes.divided_dataset.CityScapesDividedDataset(
        cityscapes.divided_dataset.Divide.Train,
        ["img", "hot_label"],
        int(patch_size),
        transform=train_transform
    )
    val_dataset = cityscapes.divided_dataset.CityScapesDividedDataset(
        cityscapes.divided_dataset.Divide.Val, 
        ["img", "hot_label", "unique_label"],
        int(patch_size), 
        transform=val_transform
    )

    return pl.LightningDataModule.from_datasets(train_dataset, batch_size=batch_size, num_workers=num_workers)
