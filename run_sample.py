import argparse
import os

from misc import pyutils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    env_group = parser.add_argument_group("Environment")
    env_group.add_argument("--voc12_root", required=True, type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
    env_group.add_argument("--num_workers", default=min(32, os.cpu_count() + 4), type=int)

    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    dataset_group.add_argument("--val_list", default="voc12/val.txt", type=str)
    dataset_group.add_argument("--infer_list", default="voc12/train.txt", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    dataset_group.add_argument("--chainer_eval_set", default="val", type=str)
    dataset_group.add_argument("--patch_size", default=32, type=int)

    cam_group = parser.add_argument_group("Class Activation Map")
    cam_group.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    cam_group.add_argument("--cam_crop_size", default=512, type=int)
    cam_group.add_argument("--cam_batch_size", default=16, type=int)
    cam_group.add_argument("--cam_num_epoches", default=5, type=int)
    cam_group.add_argument("--cam_learning_rate", default=0.1, type=float)
    cam_group.add_argument("--cam_weight_decay", default=1e-4, type=float)
    cam_group.add_argument("--cam_eval_thres", default=0.15, type=float)
    cam_group.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")

    mining_ir_group = parser.add_argument_group("Mining Inter-pixel Relations")
    mining_ir_group.add_argument("--conf_fg_thres", default=0.30, type=float)
    mining_ir_group.add_argument("--conf_bg_thres", default=0.05, type=float)

    irn_group = parser.add_argument_group("Inter-pixel Relation Network (IRNet)")
    irn_group.add_argument("--irn_network", default="net.resnet50_irn", type=str)
    irn_group.add_argument("--irn_crop_size", default=512, type=int)
    irn_group.add_argument("--irn_batch_size", default=32, type=int)
    irn_group.add_argument("--irn_num_epoches", default=3, type=int)
    irn_group.add_argument("--irn_learning_rate", default=0.1, type=float)
    irn_group.add_argument("--irn_weight_decay", default=1e-4, type=float)

    rw_params_group = parser.add_argument_group("Random Walk Params")
    rw_params_group.add_argument("--beta", default=10)
    rw_params_group.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    rw_params_group.add_argument("--ins_seg_bg_thres", default=0.25)
    rw_params_group.add_argument("--sem_seg_bg_thres", default=0.25)

    output_path_group = parser.add_argument_group("Output Path")
    output_path_group.add_argument("--log_name", default="sample_train_eval", type=str)
    output_path_group.add_argument("--cam_weights_name", default="sess/res50_cam.pth", type=str)
    output_path_group.add_argument("--irn_weights_name", default="sess/res50_irn.pth", type=str)
    output_path_group.add_argument("--cam_out_dir", default="result/cam", type=str)
    output_path_group.add_argument("--ir_label_out_dir", default="result/ir_label", type=str)
    output_path_group.add_argument("--sem_seg_out_dir", default="result/sem_seg", type=str)
    output_path_group.add_argument("--ins_seg_out_dir", default="result/ins_seg", type=str)

    step_group = parser.add_argument_group("Step")
    step_group.add_argument("--train_cam", action='store_true')
    step_group.add_argument("--make_cam", action='store_true')
    step_group.add_argument("--eval_cam", action='store_true')
    step_group.add_argument("--cam_to_ir_label", action='store_true')
    step_group.add_argument("--train_irn", action='store_true')
    step_group.add_argument("--make_ins_seg", action='store_true')
    step_group.add_argument("--eval_ins_seg", action='store_true')
    step_group.add_argument("--make_sem_seg", action='store_true')
    step_group.add_argument("--eval_sem_seg", action='store_true')

    args = parser.parse_args()

    os.makedirs("sess", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    os.makedirs(args.ins_seg_out_dir, exist_ok=True)

    pyutils.Logger(args.log_name + '.log')
    print(vars(args))

    if False:
        import step.divide_to_patches
        step.divide_to_patches.run()

    if args.train_cam is True:
        import step.train_cam_grid

        timer = pyutils.Timer('step.train_cam:')
        step.train_cam_grid.run(args)

    if args.make_cam is True:
        import step.make_cam

        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args)

    if args.eval_cam is True:
        import step.eval_cam

        timer = pyutils.Timer('step.eval_cam:')
        step.eval_cam.run(args)

    if args.cam_to_ir_label is True:
        import step.cam_to_ir_label

        timer = pyutils.Timer('step.cam_to_ir_label:')
        step.cam_to_ir_label.run(args)

    if args.train_irn is True:
        import step.train_irn

        timer = pyutils.Timer('step.train_irn:')
        step.train_irn.run(args)

    if args.make_ins_seg is True:
        import step.make_ins_seg_labels

        timer = pyutils.Timer('step.make_ins_seg_labels:')
        step.make_ins_seg_labels.run(args)

    if args.eval_ins_seg is True:
        import step.eval_ins_seg

        timer = pyutils.Timer('step.eval_ins_seg:')
        step.eval_ins_seg.run(args)

    if args.make_sem_seg is True:
        import step.make_sem_seg_labels

        timer = pyutils.Timer('step.make_sem_seg_labels:')
        step.make_sem_seg_labels.run(args)

    if args.eval_sem_seg is True:
        import step.eval_sem_seg

        timer = pyutils.Timer('step.eval_sem_seg:')
        step.eval_sem_seg.run(args)

