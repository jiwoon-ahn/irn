import numpy as np
import voc12.dataloader
from torch.utils.data import DataLoader
from pycococreatortools import pycococreatortools
import os
import json

VOC2012_JSON_FOLDER = ""

def run(args):

    infer_dataset = voc12.dataloader.VOC12ImageDataset(args.infer_list, voc12_root=args.voc12_root)

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    val_json = json.load(open(os.path.join(VOC2012_JSON_FOLDER, 'pascal_val2012.json')))
    # Do not use this file for evaluation!

    coco_output = {}
    coco_output["images"] = []
    coco_output["annotations"] = []
    coco_output['categories'] = val_json['categories']
    coco_output['type'] = val_json['type']

    for iter, pack in enumerate(infer_data_loader):

        img_name = pack['name'][0]
        img_id = int(img_name[:4] + img_name[5:])
        img_size = pack['img'].shape[2:]

        image_info = pycococreatortools.create_image_info(
            img_id, img_name + ".jpg", (img_size[1], img_size[0]))
        coco_output["images"].append(image_info)
        ann = np.load(os.path.join(args.ins_seg_out_dir, img_name) + '.npy', allow_pickle=True).item()

        instance_id = 1

        for score, mask, class_id in zip( ann['score'], ann['mask'], ann['class']):
            if score < 1e-5:
                continue
            category_info = {'id': class_id, 'is_crowd': False}

            annotation_info = pycococreatortools.create_annotation_info(
                instance_id, img_id, category_info, mask, img_size[::-1], tolerance=0)
            instance_id += 1
            coco_output['annotations'].append(annotation_info)

    with open('voc2012_train_custom.json', 'w') as outfile:
        json.dump(coco_output, outfile)


