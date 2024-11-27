import argparse
import os
import shutil
import warnings

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
from data.mvtec_dataset import MVTecDataset
from eval import evaluate
from model.destseg import DeSTSeg
from model.losses import cosine_similarity_loss, focal_loss, l1_loss

warnings.filterwarnings("ignore")


def str2bool(v):
    """
    Converts a string to a boolean value.
    
    Input:
        v - string
    Output:
        True/False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train(args, category, rotate_90=False, random_rotate=0):
    # Create necessary directories
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    run_name = f"{args.run_name_head}_{args.steps}_{category}"
    run_log_path = os.path.join(args.log_path, run_name)

    if os.path.exists(run_log_path):
        shutil.rmtree(run_log_path)

    visualizer = SummaryWriter(log_dir=run_log_path)

    # Initialize model and optimizers
    model = DeSTSeg(dest=True, ed=True).cuda()

    seg_optimizer = torch.optim.SGD(
        [
            {"params": model.segmentation_net.res.parameters(), "lr": args.lr_res},
            {"params": model.segmentation_net.head.parameters(), "lr": args.lr_seghead},
        ],
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )
    seg_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        seg_optimizer, mode='min', factor=0.1, patience=10, verbose=True
    )

    de_st_optimizer = torch.optim.SGD(
        [
            {"params": model.student_net.parameters(), "lr": args.lr_de_st},
        ],
        lr=0.4,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )
    de_st_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        de_st_optimizer, mode='min', factor=0.1, patience=10, verbose=True
    )

    dataset = MVTecDataset(
        is_train=True,
        mvtec_dir=os.path.join(args.mvtec_path, category, "train/good/"),
        resize_shape=RESIZE_SHAPE,
        normalize_mean=NORMALIZE_MEAN,
        normalize_std=NORMALIZE_STD,
        dtd_dir=args.dtd_path,
        rotate_90=rotate_90,
        random_rotate=random_rotate,
        additional_augmentation=False  
    )
    print(len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    global_step = 0
    best_performance = float('-inf')
    flag = True

    while flag:
        for _, sample_batched in enumerate(dataloader):
            seg_optimizer.zero_grad()
            de_st_optimizer.zero_grad()

            img_origin = sample_batched["img_origin"].cuda()
            img_aug = sample_batched["img_aug"].cuda()
            mask = sample_batched["mask"].cuda()

            if global_step < args.de_st_steps:
                model.student_net.train()
                model.segmentation_net.eval()
            else:
                model.student_net.eval()
                model.segmentation_net.train()

            fp = 0.3 if args.fp else 0

            # Forward pass
            output_segmentation, output_de_st, output_de_st_list, loss_reviewkd = model(
                img_aug, img_origin, fp
            )

            # Prepare mask for loss calculation
            mask = F.interpolate(
                mask,
                size=output_segmentation.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.where(mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask))

            # Calculate losses
            cosine_loss_val = cosine_similarity_loss(output_de_st_list)
            focal_loss_val = focal_loss(output_segmentation, mask, gamma=args.gamma)
            l1_loss_val = l1_loss(output_segmentation, mask)

            # Backpropagation
            if global_step < args.de_st_steps:
                if args.kr:
                    total_loss_val = cosine_loss_val + loss_reviewkd
                    total_loss_val.backward()
                    de_st_optimizer.step()
                else:
                    total_loss_val = cosine_loss_val
                    total_loss_val.backward()
                    de_st_optimizer.step()
            else:
                total_loss_val = focal_loss_val + l1_loss_val
                total_loss_val.backward()
                seg_optimizer.step()

            # Logging
            global_step += 1
            visualizer.add_scalar("cosine_loss", cosine_loss_val, global_step)
            visualizer.add_scalar("focal_loss", focal_loss_val, global_step)
            visualizer.add_scalar("l1_loss", l1_loss_val, global_step)
            visualizer.add_scalar("total_loss", total_loss_val, global_step)

            # Evaluation and model saving
            if global_step % args.eval_per_steps == 0:
                val = evaluate(args, category, model, visualizer, global_step)
                if val > best_performance:
                    best_performance = val
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.checkpoint_path, f"{run_name}_best.pckl")
                    )

            # Print progress
            if global_step % args.log_per_steps == 0:
                loss_info = (
                    f"Training at global step {global_step}, "
                    f"cosine loss: {round(float(cosine_loss_val), 4)}"
                    if global_step < args.de_st_steps
                    else f"Training at global step {global_step}, "
                         f"focal loss: {round(float(focal_loss_val), 4)}, "
                         f"l1 loss: {round(float(l1_loss_val), 4)}"
                )
                print(loss_info)

            # Terminate training if steps reached
            if global_step >= args.steps:
                flag = False
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--mvtec_path", type=str, default="./datasets/under_vehicle/")
    parser.add_argument("--dtd_path", type=str, default="./datasets/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_model/")
    parser.add_argument("--run_name_head", type=str, default="DeSTSeg_UnderVehicle")
    parser.add_argument("--log_path", type=str, default="./logs/")
    parser.add_argument('--fp', default=True, type=str2bool)
    parser.add_argument('--kr', default=True, type=str2bool)



    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr_de_st", type=float, default=0.4)
    parser.add_argument("--lr_res", type=float, default=0.1)
    parser.add_argument("--lr_seghead", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--de_st_steps", type=int, default=1000)  # steps of training the denoising student model
    parser.add_argument("--eval_per_steps", type=int, default=1000)
    parser.add_argument("--log_per_steps", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=4)  # for focal loss
    parser.add_argument("--T", type=int, default=100)  # for image-level inference

    parser.add_argument("--custom_training_category", action="store_true", default=False)
    parser.add_argument("--no_rotation_category", nargs="*", type=str, default=list())
    parser.add_argument("--slight_rotation_category", nargs="*", type=str, default=list())
    parser.add_argument("--rotation_category", nargs="*", type=str, default=list())

    args = parser.parse_args()

    if args.custom_training_category:
        no_rotation_category = args.no_rotation_category
        slight_rotation_category = args.slight_rotation_category
        rotation_category = args.rotation_category

        # Check categories
        for category in (no_rotation_category + slight_rotation_category + rotation_category):
            assert category in ALL_CATEGORY
    else:
        no_rotation_category = [
            "Chevrolet", "Generated_3", "Generated_6", "KIA_Rio", "Camry",
            "Generated_2", "Generated_5", "KIA_Pegas", "BMW", "Generated_1",
            "Generated_4", "Hyundai","Toyota_Innova","Camry2", "CamryToday"
        ]

        slight_rotation_category = [ "Camry"  # Categories that need slight rotation
    ]

    with torch.cuda.device(args.gpu_id):
        for obj in no_rotation_category:
            print(obj)
            train(args, obj)

        for obj in slight_rotation_category:
            print(obj)
            train(args, obj, rotate_90=False, random_rotate=5)

        for obj in rotation_category:
            print(obj)
            train(args, obj, rotate_90=True, random_rotate=5)
