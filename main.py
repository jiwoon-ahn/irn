import argparse
import voc12.dataloader
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", default='./voc12/train_aug.txt', type=str)
    parser.add_argument("--val_list", default='./voc12/val.txt', type=str)
    parser.add_argument("--out", default="./voc12/cls_labels_divided.npy", type=str)
    parser.add_argument("--voc12_root", default="/home/postech2/irn/VOCdevkit/VOC2012", type=str)
    parser.add_argument("--divide_factor", default=3, type=int)
    args = parser.parse_args()

    train_name_list = voc12.dataloader.load_img_name_list(args.train_list)

    for name in tqdm(train_name_list):
        voc12.dataloader.save_divided_image_and_label_from_xml(name, args.divide_factor, args.voc12_root)