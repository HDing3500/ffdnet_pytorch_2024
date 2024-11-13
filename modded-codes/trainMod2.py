import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import FFDNet
from dataset import Dataset
from utils import weights_init_kaiming, batch_psnr, init_logger
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import re


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Define the extract_identifier function globally
def extract_identifier(filename):
    """
    Extracts a unique identifier from the filename. Assumes filename format like:
    '0001_GT_SRGB_001.PNG' or '0001_NOISY_SRGB_001.PNG'.
    """
    match = re.match(r'(\d{4})_(NOISY|GT)_SRGB_(\d{3})\.PNG', filename)
    if match:
        return match.group(1) + '_' + match.group(3)
    else:
        return None


class CustomDataset(Dataset):
    def __init__(self, gt_folder, noisy_folder, transform=None):
        self.gt_folder = gt_folder
        self.noisy_folder = noisy_folder
        self.transform = transform

        # Get all image file paths
        self.gt_images = [os.path.join(gt_folder, f) for f in os.listdir(gt_folder) if f.endswith('.PNG')]
        self.noisy_images = [os.path.join(noisy_folder, f) for f in os.listdir(noisy_folder) if f.endswith('.PNG')]

        # Extract common identifiers and filter accordingly
        gt_dict = {extract_identifier(os.path.basename(f)): f for f in self.gt_images}
        noisy_dict = {extract_identifier(os.path.basename(f)): f for f in self.noisy_images}
        common_keys = set(gt_dict.keys()) & set(noisy_dict.keys())

        self.gt_images = [gt_dict[key] for key in common_keys]
        self.noisy_images = [noisy_dict[key] for key in common_keys]

    def __len__(self):
        return len(self.gt_images)

    def __getitem__(self, idx):
        # Dynamically load the images only when needed
        gt_img = Image.open(self.gt_images[idx]).convert('RGB')
        noisy_img = Image.open(self.noisy_images[idx]).convert('RGB')

        # Verify initial shape of images
        print(f'GT image shape: {gt_img.size}')
        print(f'Noisy image shape: {noisy_img.size}')

        if self.transform:
            gt_img = self.transform(gt_img)
            noisy_img = self.transform(noisy_img)

        # After transformation, the shape should be [C, H, W] (e.g., [3, H, W] for RGB images)
        print(f'GT image tensor shape after transform: {gt_img.shape}')
        print(f'Noisy image tensor shape after transform: {noisy_img.shape}')

        # Check if noisy_img has 3 channels (RGB) and gt_img has 3 channels (RGB)
        assert noisy_img.shape[0] == 3, f"Expected 3 channels in noisy image, got {noisy_img.shape[0]}"
        assert gt_img.shape[0] == 3, f"Expected 3 channels in GT image, got {gt_img.shape[0]}"

        # Generate a noise map (for example, random noise for simplicity)
        noise_map = torch.randn_like(noisy_img[0:1, :, :])  # Generate noise map with same height and width as the noisy image

        # Verify the noise map shape
        print(f'Noise map shape: {noise_map.shape}')

        # If it's a noisy image, concatenate the noise map along the channel dimension
        noisy_img = torch.cat([noisy_img, noise_map], dim=0)  # Concatenate along the channel axis (C)
        
        # Verify the final shape after concatenation
        print(f'Noisy image shape after concatenation: {noisy_img.shape}')
        # Expecting: [4, H, W] (3 RGB channels + 1 noise map channel)

        # Return the noisy image with 4 channels and the ground truth image (3 channels)
        return noisy_img, gt_img




    



def load_data(gt_folder, noisy_folder, split_ratio=0.8):
    # Get all the .png files in each folder
    gt_images = [os.path.join(gt_folder, f) for f in os.listdir(gt_folder) if f.endswith('.PNG')]
    noisy_images = [os.path.join(noisy_folder, f) for f in os.listdir(noisy_folder) if f.endswith('.PNG')]

    # Create a dictionary with identifiers as keys and file paths as values for both noisy and ground truth images
    gt_dict = {extract_identifier(os.path.basename(f)): f for f in gt_images}
    noisy_dict = {extract_identifier(os.path.basename(f)): f for f in noisy_images}

    # Find common identifiers between ground truth and noisy images
    common_keys = set(gt_dict.keys()) & set(noisy_dict.keys())

    # Make sure that we have a matching set of noisy and ground truth images
    gt_images = [gt_dict[key] for key in common_keys]
    noisy_images = [noisy_dict[key] for key in common_keys]

    # Define the transformation (resize and convert to tensor)
    transform = transforms.Compose([
        transforms.Resize((256, 192)),  # Optional resizing to reduce memory
        transforms.ToTensor()
    ])

    # Initialize datasets with transformations
    dataset_train = CustomDataset(gt_folder, noisy_folder, transform)
    dataset_val = CustomDataset(gt_folder, noisy_folder, transform)  # For validation

    # Split data into training and validation sets
    split_index = int(len(dataset_train) * split_ratio)
    dataset_train, dataset_val = torch.utils.data.random_split(dataset_train, [split_index, len(dataset_train) - split_index])

    return dataset_train, dataset_val


def main(args, train_data, val_data):
    loader_train = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True)
    loader_val = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    writer = SummaryWriter(args.log_dir)
    logger = init_logger(args)

    # Initialize the model
    in_ch = 4  # Assuming RGB + noise map
    net = FFDNet(num_input_channels=in_ch)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(net).to(device)
    criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0  # Initialize start_epoch here

    # Resume training if specified
    if args.resume_training:
        resumef = os.path.join(args.log_dir, 'ckpt.pth')
        if os.path.isfile(resumef):
            checkpoint = torch.load(resumef)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args = checkpoint['args']
            start_epoch = checkpoint['training_params']['start_epoch']  # Resume from the saved epoch
        else:
            print(f"Checkpoint not found at {resumef}, starting training from scratch.")

    scaler = torch.cuda.amp.GradScaler()  # For mixed precision

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for i, (img_train, _) in enumerate(loader_train):
            optimizer.zero_grad()
            img_train = img_train.cuda()

            # Generate a dummy noise map (zeros) with the same batch size as img_train
            dummy_noise_map = torch.zeros_like(img_train[:, :1, :, :])  # [batch_size, 1, H, W]

            # Concatenate the dummy noise map to the ground truth to make it a 4-channel image
            img_train_4chan = torch.cat([img_train, dummy_noise_map], dim=1)  # [batch_size, 4, H, W]

            with torch.cuda.amp.autocast():  # Enable mixed precision
                out_train = model(img_train)  # Model processes the noisy image

                # Ensure the output is resized to the same size as the input (img_train)
                out_train = torch.nn.functional.interpolate(out_train, size=img_train.shape[2:], mode='bilinear', align_corners=False)

                # Now calculate the loss
                loss = criterion(out_train, img_train_4chan) / (img_train.size(0) * 2)  # Divide by batch size (or another appropriate factor)

            scaler.scale(loss).backward()  # Scale loss for backward pass
            scaler.step(optimizer)
            scaler.update()

            # Additional logging for PSNR or other metrics if needed
            if i % args.save_every == 0:
                psnr_train = batch_psnr(out_train, img_train, 1.)
                writer.add_scalar('loss', loss.item(), epoch * len(loader_train) + i)
                writer.add_scalar('PSNR on training data', psnr_train, epoch * len(loader_train) + i)
                print(f"[epoch {epoch+1}][{i+1}/{len(loader_train)}] loss: {loss.item():.4f} PSNR_train: {psnr_train:.4f}")

            torch.cuda.empty_cache()  # Clear cache

        # Validation
        psnr_val = 0
        model.eval()
        with torch.no_grad():
            for valimg in loader_val:
                img_val = valimg[0].cuda()  # Noisy image on GPU (use noisy images directly)
                out_val = model(img_val)  # Model processes noisy image
                out_val = torch.clamp(out_val, 0., 1.)  # Clamping the output
                psnr_val += batch_psnr(out_val, img_val, 1.)

        psnr_val /= len(loader_val)  # Average PSNR for validation
        print(f"\n[epoch {epoch+1}] PSNR_val: {psnr_val:.4f}")
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)

        # Save model and checkpoint
        torch.save(model.state_dict(), os.path.join(args.log_dir, 'net.pth'))
        if epoch % args.save_every_epochs == 0:
            torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.log_dir, f'ckpt_e{epoch+1}.pth'))







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFDNet")
    parser.add_argument("--gray", action='store_true', help='Train with grayscale images.')
    parser.add_argument("--log_dir", type=str, default="logs", help='Path of log files')
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")  # Reduced batch size
    parser.add_argument("--epochs", "--e", type=int, default=80, help="Number of total training epochs")
    parser.add_argument("--resume_training", "--r", action='store_true', help="Resume training from a previous checkpoint")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--save_every", type=int, default=10, help="Number of training steps to log PSNR")
    parser.add_argument("--save_every_epochs", type=int, default=5, help="Number of epochs after which to save checkpoints")
    args = parser.parse_args()

    # Define paths to your dataset
    gt_folder = r"C:\Users\dingh\Desktop\Indigo\ffdnet-pytorch - Copy\GT"
    noisy_folder = r"C:\Users\dingh\Desktop\Indigo\ffdnet-pytorch - Copy\NOISY"

    # Load data
    dataset_train, dataset_val = load_data(gt_folder, noisy_folder)
    
    # Run the main function
    main(args, dataset_train, dataset_val)
