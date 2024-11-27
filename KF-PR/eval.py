import argparse
import os
import shutil
import warnings
from typing import Dict, Tuple, List
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics import AUROC, AveragePrecision
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score, roc_curve
from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
from data.mvtec_dataset import MVTecDataset
from model.destseg import DeSTSeg
from model.metrics import AUPRO, IAPS
class MetricTracker:
    """Tracks all metrics during evaluation"""
    def __init__(self, device='cuda'):
        self.device = device
        # Initialize torchmetrics for all images
        self.pixel_auroc = AUROC(task='binary').to(device)
        self.pixel_ap = AveragePrecision(task='binary').to(device)
        self.pixel_aupro = AUPRO().to(device)
        self.pixel_iaps = IAPS().to(device)
        self.image_ap = AveragePrecision(task='binary').to(device)
        
        # Initialize separate metrics for anomaly images only
        self.anomaly_pixel_auroc = AUROC(task='binary').to(device)
        self.anomaly_pixel_ap = AveragePrecision(task='binary').to(device)
        self.anomaly_pixel_aupro = AUPRO().to(device)
        self.anomaly_pixel_iaps = IAPS().to(device)
        
        # Storage for manual metrics
        self.image_scores = []
        self.image_labels = []
        self.pixel_scores = []
        self.pixel_labels = []
        
        # Counters
        self.total_good_images = 0
        self.total_anomaly_images = 0
        self.false_positives = 0
        self.false_negatives = 0

    def compute_per_image_metrics(self, output: torch.Tensor, mask: torch.Tensor) -> Dict:
        """Compute metrics for a single image"""
        # Initialize per-image metrics
        metrics = {}
        
        # Compute AUROC for single image
        auroc = AUROC(task='binary').to(self.device)
        auroc.update(output.flatten(), mask.flatten())
        metrics['auroc'] = float(auroc.compute())
        
        # Compute AP for single image
        ap = AveragePrecision(task='binary').to(self.device)
        ap.update(output.flatten(), mask.flatten())
        metrics['ap'] = float(ap.compute())
        
        # Compute PRO for single image
        aupro = AUPRO().to(self.device)
        aupro.update(output, mask)
        metrics['pro'] = float(aupro.compute())
        
        # Compute IAP and IAP90 for single image
        iaps = IAPS().to(self.device)
        iaps.update(output, mask)
        iap, iap90 = iaps.compute()
        metrics['iap'] = float(iap)
        metrics['iap90'] = float(iap90)
        
        return metrics
        
    def update(self, output: torch.Tensor, mask: torch.Tensor, image_score: torch.Tensor, is_good: bool):
        """Update all metrics with new batch"""
        # Flatten predictions and targets for pixel-level metrics
        pred_flat = output.view(-1)
        target_flat = mask.view(-1)
        
        # Update overall metrics
        self.pixel_auroc.update(pred_flat, target_flat)
        self.pixel_ap.update(pred_flat, target_flat)
        self.pixel_aupro.update(output, mask)
        self.pixel_iaps.update(output, mask)
        
        # Update image-level AP
        self.image_ap.update(image_score, torch.tensor([0 if is_good else 1], device=self.device))
        
        # Update anomaly-only metrics if this is an anomaly image
        if not is_good:
            self.anomaly_pixel_auroc.update(pred_flat, target_flat)
            self.anomaly_pixel_ap.update(pred_flat, target_flat)
            self.anomaly_pixel_aupro.update(output, mask)
            self.anomaly_pixel_iaps.update(output, mask)
        
        # Store scores for manual metrics
        self.pixel_scores.extend(pred_flat.cpu().numpy())
        self.pixel_labels.extend(target_flat.cpu().numpy())
        self.image_scores.extend(image_score.cpu().numpy())
        self.image_labels.extend([0 if is_good else 1])
        
        # Update counters
        if is_good:
            self.total_good_images += 1
        else:
            self.total_anomaly_images += 1

    def compute_metrics(self, threshold: float) -> Dict:
        """Compute all final metrics"""
        # Image-level predictions
        image_preds = (np.array(self.image_scores) >= threshold).astype(int)
        
        # Update error counters
        self.false_positives = sum((image_preds == 1) & (np.array(self.image_labels) == 0))
        self.false_negatives = sum((image_preds == 0) & (np.array(self.image_labels) == 1))
        
        # Calculate metrics
        metrics = {
            'pixel_auroc': float(self.pixel_auroc.compute()),
            'pixel_ap': float(self.pixel_ap.compute()),
            'pixel_aupro': float(self.pixel_aupro.compute()),
            'pixel_iap': float(self.pixel_iaps.compute()[0]),
            'pixel_iap90': float(self.pixel_iaps.compute()[1]),
            'image_auroc': roc_auc_score(self.image_labels, self.image_scores),
            'image_ap': float(self.image_ap.compute()),
            'image_accuracy': accuracy_score(self.image_labels, image_preds),
            'image_precision': precision_score(self.image_labels, image_preds),
            'image_recall': recall_score(self.image_labels, image_preds),
            'image_f1': f1_score(self.image_labels, image_preds)
        }
        
        # Add anomaly-only metrics if we have anomaly images
        if self.total_anomaly_images > 0:
            metrics.update({
                'anomaly_pixel_auroc': float(self.anomaly_pixel_auroc.compute()),
                'anomaly_pixel_ap': float(self.anomaly_pixel_ap.compute()),
                'anomaly_pixel_aupro': float(self.anomaly_pixel_aupro.compute()),
                'anomaly_pixel_iap': float(self.anomaly_pixel_iaps.compute()[0]),
                'anomaly_pixel_iap90': float(self.anomaly_pixel_iaps.compute()[1])
            })
        
        return metrics

def denormalize(image: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    """Denormalize image for visualization"""
    image = (image * np.array(std)) + np.array(mean)
    return np.clip(image, 0, 1)

def visualize_predictions(
    image: torch.Tensor,
    mask: torch.Tensor,
    prediction: torch.Tensor,
    idx: int,
    global_step: int,
    save_dir: str = '/content/evaluation_plots'
):
    """Create and save visualization of predictions"""
    # Prepare image
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = denormalize(image_np, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # Prepare prediction
    prediction_np = prediction.squeeze().cpu().numpy()
    prediction_np = (prediction_np - prediction_np.min()) / (prediction_np.max() - prediction_np.min())
    prediction_np = cv2.resize(prediction_np, (image_np.shape[1], image_np.shape[0]))
    
    # Prepare mask
    mask_np = mask.squeeze().cpu().numpy()
    binary_mask_np = (prediction_np >= 0.5).astype(np.uint8)
    
    # Create visualization
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    
    # Original image
    axs[0].imshow(image_np)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    
    # Ground truth mask
    axs[1].imshow(mask_np, cmap='gray')
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis("off")
    
    # # Prediction heatmap
    # im = axs[2].imshow(prediction_np, cmap='jet')
    # axs[2].set_title("Prediction")
    # axs[2].axis("off")
    # fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    
    # Binary prediction
    axs[2].imshow(binary_mask_np, cmap='gray')
    axs[2].set_title("Binary Mask (Pred)")
    axs[2].axis("off")
    
    # Overlay
    axs[3].imshow(image_np)
    im = axs[3].imshow(prediction_np, cmap='jet', alpha=0.5)
    axs[3].set_title("Heatmap Overlay")
    axs[3].axis("off")
    fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/plot_{idx}_{global_step}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_image_score(segmentation_output: torch.Tensor, T: int) -> torch.Tensor:
    """Calculate image-level score using top-T pixels"""
    output_sorted, _ = torch.sort(
        segmentation_output.view(segmentation_output.size(0), -1),
        dim=1,
        descending=True,
    )
    return torch.mean(output_sorted[:, :T], dim=1)



def evaluate(args, category: str, model: torch.nn.Module, visualizer: SummaryWriter, global_step: int = 0, test: bool = False,saved_threshold: float = 0.5) -> float:
    """Main evaluation function"""
    model.eval()
    
    # Initialize metric tracker
    tracker = MetricTracker()

    dataset_path = os.path.join(args.mvtec_path, category, "test" if test else "val")
    dataset = MVTecDataset(
        is_train=False,
        mvtec_dir=dataset_path,
        resize_shape=RESIZE_SHAPE,
        normalize_mean=NORMALIZE_MEAN,
        normalize_std=NORMALIZE_STD,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    optimal_threshold = 0.5
    print(f"Dataset size: {len(dataset)}")
    
    with torch.no_grad():
        for idx, sample_batched in enumerate(dataloader):
            # Get batch data
            img = sample_batched["img"].cuda()
            mask = sample_batched["mask"].cuda()
            is_good = torch.max(mask) == 0
            
            # Get model predictions
            output_segmentation, _, _, _ = model(img)
            
            # Resize output to match mask size
            output_segmentation = F.interpolate(
                output_segmentation,
                size=mask.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            
            # Calculate image-level score
            image_score = calculate_image_score(output_segmentation, args.T)
            
             # Update overall metrics
            tracker.update(output_segmentation, mask, image_score, is_good)

            print(f"\n=== Image {idx + 1} ===")
            print(f"Image-Level Score: {float(image_score):.4f} ")
            
            if test and optimal_threshold is not None:
                prediction = "Anomaly" if float(image_score) >= optimal_threshold else "Good"
                is_correct = (prediction == "Anomaly") != is_good
                print(f"Ground Truth: {'Good' if is_good else 'Anomaly'}")
                print(f"Prediction: {prediction} ({'Correct' if is_correct else 'Incorrect'})")
            # For anomaly images, calculate and print per-image segmentation metrics
            if not is_good:
                metrics = tracker.compute_per_image_metrics(output_segmentation, mask)
                print("Pixel-Level Metrics:")
                print(f"  AUROC: {metrics['auroc']:.4f} AP: {metrics['ap']:.4f} PRO: {metrics['pro']:.4f} IAP: {metrics['iap']:.4f} IAP90: {metrics['iap90']:.4f}")
            
           
            
            # Visualize predictions
            visualize_predictions(img[0], mask[0], output_segmentation[0], idx + 1, global_step)
    
  

    if test:
        optimal_threshold = 0.5

    metrics = tracker.compute_metrics(optimal_threshold)
    
    # Print final results
    print(f"\nResults with optimal threshold ({optimal_threshold:.4f}):")
    print("\nGood Images Statistics:")
    if tracker.total_good_images > 0:
        fp_rate = tracker.false_positives / tracker.total_good_images * 100
        print(f"Total good images: {tracker.total_good_images}")
        print(f"False positive rate: {fp_rate:.2f}%")
        print(f"False positives: {tracker.false_positives}")
    
    print("\nAnomaly Images Statistics:")
    if tracker.total_anomaly_images > 0:
        fn_rate = tracker.false_negatives / tracker.total_anomaly_images * 100
        print(f"Total anomaly images: {tracker.total_anomaly_images}")
        print(f"False negative rate: {fn_rate:.2f}%")
        print(f"Number of anomaly images missed: {tracker.false_negatives}")
        print("\nAnomaly-Only Pixel-Level Metrics:")
        print(f"Anomaly Pixel-AUC: {metrics['anomaly_pixel_auroc']:.4f}")
        print(f"Anomaly Pixel-AP: {metrics['anomaly_pixel_ap']:.4f}")
        print(f"Anomaly Pixel-PRO: {metrics['anomaly_pixel_aupro']:.4f}")
        print(f"Anomaly Pixel-IAP: {metrics['anomaly_pixel_iap']:.4f}")
        print(f"Anomaly Pixel-IAP90: {metrics['anomaly_pixel_iap90']:.4f}")
    
    print("\nOverall Pixel-Level Metrics:")
    print(f"AUROC: {metrics['pixel_auroc']:.4f}")
    print(f"AP: {metrics['pixel_ap']:.4f}")
    print(f"PRO: {metrics['pixel_aupro']:.4f}")
    print(f"IAP: {metrics['pixel_iap']:.4f}")
    print(f"IAP90: {metrics['pixel_iap90']:.4f}")
    
    print("\nImage-Level Metrics:")
    print(f"AUROC: {metrics['image_auroc']:.4f}")
    print(f"AP: {metrics['image_ap']:.4f}")  # Added image-level AP
    print(f"Accuracy: {metrics['image_accuracy']:.4f}")
    print(f"Precision: {metrics['image_precision']:.4f}")
    print(f"Recall: {metrics['image_recall']:.4f}")
    print(f"F1 Score: {metrics['image_f1']:.4f}")
    
    # Log metrics to visualizer
    if visualizer is not None:
        visualizer.add_scalar(f'{category}/pixel_auroc', metrics['pixel_auroc'], global_step)
        visualizer.add_scalar(f'{category}/pixel_ap', metrics['pixel_ap'], global_step)
        visualizer.add_scalar(f'{category}/pixel_aupro', metrics['pixel_aupro'], global_step)
        visualizer.add_scalar(f'{category}/pixel_iap', metrics['pixel_iap'], global_step)
        visualizer.add_scalar(f'{category}/pixel_iap90', metrics['pixel_iap90'], global_step)
        visualizer.add_scalar(f'{category}/image_auroc', metrics['image_auroc'], global_step)
        visualizer.add_scalar(f'{category}/image_ap', metrics['image_ap'], global_step)  # Added image-level AP logging
        visualizer.add_scalar(f'{category}/image_accuracy', metrics['image_accuracy'], global_step)
        visualizer.add_scalar(f'{category}/image_f1', metrics['image_f1'], global_step)
    
    # Calculate final metric value (including image-level AP in the average)
    valid_metrics = [
        metrics['pixel_auroc'],
        metrics['pixel_ap'],
        metrics['pixel_aupro'],
        metrics['image_auroc'],
        metrics['image_ap'],  # Added image-level AP to final metric
        metrics['pixel_iap'],
        metrics['pixel_iap90']
    ]
    return sum(m for m in valid_metrics if not (np.isnan(m) or np.isinf(m)))


def test(args, category: str):
    """Test function"""
    # Setup logging
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    run_name = f"DeSTSeg_UnderVehicle_test_{category}"
    log_dir = os.path.join(args.log_path, run_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    
    visualizer = SummaryWriter(log_dir=log_dir)
    
    # Setup model
    model = DeSTSeg(dest=True, ed=True).cuda()
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_path, f"{args.base_model_name}{category}_best6.pckl")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    model.load_state_dict(torch.load(checkpoint_path))
    evaluate(args, category, model, visualizer, test=True,saved_threshold=0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--mvtec_path", type=str, default="./datasets/under_vehicle/")
    parser.add_argument("--dtd_path", type=str, default="./datasets/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="./expermintes ( threshold = 0.5 )/")
    parser.add_argument("--base_model_name", type=str, default="DeSTSeg_UnderVehicle_9000_")
    parser.add_argument("--log_path", type=str, default="./logs/")
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--category", nargs="*", type=str, default=ALL_CATEGORY)
    
    args = parser.parse_args()
    
    # Validate categories
    for obj in args.category:
        assert obj in ALL_CATEGORY, f"Invalid category: {obj}"
    
    # Run test for each category
    with torch.cuda.device(args.gpu_id):
        for obj in args.category:
            print(f"{obj}")
            test(args, obj)