import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as utils
from tensorboardX import SummaryWriter
from models import FFDNet
from dataset import Dataset
from utils import weights_init_kaiming, batch_psnr, init_logger
from torchvision import transforms
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class CustomDataset(Dataset):
    def __init__(self, gt_tensor, noisy_tensor):
        self.gt_tensor = gt_tensor
        self.noisy_tensor = noisy_tensor

    def __len__(self):
        return len(self.gt_tensor)

    def __getitem__(self, idx):
        return self.noisy_tensor[idx], self.gt_tensor[idx]

def load_data(gt_folder, noisy_folder, split_ratio=0.8):
    gt_images = [os.path.join(gt_folder, f) for f in os.listdir(gt_folder) if f.endswith('.png')]
    noisy_images = [os.path.join(noisy_folder, f) for f in os.listdir(noisy_folder) if f.endswith('.png')]
    
    assert len(gt_images) == len(noisy_images), "Mismatch between ground truth and noisy images."
    
    transform = transforms.ToTensor()
    gt_tensors = []
    noisy_tensors = []

    for gt_path, noisy_path in zip(gt_images, noisy_images):
        gt_img = Image.open(gt_path).convert('RGB')
        noisy_img = Image.open(noisy_path).convert('RGB')
        gt_tensors.append(transform(gt_img))
        noisy_tensors.append(transform(noisy_img))

    gt_tensor = torch.stack(gt_tensors)
    noisy_tensor = torch.stack(noisy_tensors)
    
    # Split data into training and validation sets
    split_index = int(len(gt_tensor) * split_ratio)
    gt_train, gt_val = gt_tensor[:split_index], gt_tensor[split_index:]
    noisy_train, noisy_val = noisy_tensor[:split_index], noisy_tensor[split_index:]
    
    return (gt_train, noisy_train), (gt_val, noisy_val)

def main(args, train_data, val_data):
    gt_tensor, noisy_tensor = train_data
    gt_val_tensor, noisy_val_tensor = val_data

    print('> Loading dataset ...')
    dataset_train = CustomDataset(gt_tensor, noisy_tensor)
    dataset_val = CustomDataset(gt_val_tensor, noisy_val_tensor)  # Initialize validation dataset
    loader_train = DataLoader(dataset=dataset_train, num_workers=6, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset=dataset_val, num_workers=6, batch_size=args.batch_size, shuffle=False)  # Create a DataLoader for validation

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(args.log_dir)
    logger = init_logger(args)

    in_ch = 1 if args.gray else 3
    net = FFDNet(num_input_channels=in_ch)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(net).to(device)
    criterion.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume_training:
        resumef = os.path.join(args.log_dir, 'ckpt.pth')
        if os.path.isfile(resumef):
            checkpoint = torch.load(resumef)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args = checkpoint['args']
            start_epoch = checkpoint['training_params']['start_epoch']
        else:
            raise Exception("Couldn't resume training with checkpoint {}".format(resumef))

    scaler = torch.cuda.amp.GradScaler()  # For mixed precision

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for i, (img_train, _) in enumerate(loader_train):
            optimizer.zero_grad()
            img_train = img_train.cuda()
            imgn_train = img_train
            noise_sigma = torch.full((imgn_train.size(0),), args.val_noiseL, device=imgn_train.device)

            with torch.cuda.amp.autocast():  # Enable mixed precision
                out_train = model(imgn_train, noise_sigma)
                loss = criterion(out_train, img_train) / (imgn_train.size(0) * 2)

            scaler.scale(loss).backward()  # Scale loss for backward pass
            scaler.step(optimizer)
            scaler.update()

            if i % args.save_every == 0:
                psnr_train = batch_psnr(out_train, img_train, 1.)
                writer.add_scalar('loss', loss.item(), epoch * len(loader_train) + i)
                writer.add_scalar('PSNR on training data', psnr_train, epoch * len(loader_train) + i)
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                      (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))

            torch.cuda.empty_cache()  # Clear cache

        # Validation
        psnr_val = 0
        model.eval()
        with torch.no_grad():
            for valimg in loader_val:
                img_val = valimg[1].cuda()  # Ground truth image on GPU
                noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=args.val_noiseL).cuda()
                imgn_val = img_val + noise

                # Create noise_sigma tensor
                noise_sigma = torch.full((img_val.size(0),), args.val_noiseL, device=img_val.device)

                out_val = model(imgn_val, noise_sigma)  # Pass noise_sigma to the model
                out_val = torch.clamp(out_val, 0., 1.)  # Clamping the output
                psnr_val += batch_psnr(out_val, img_val, 1.)

        psnr_val /= len(loader_val)  # Average PSNR for validation
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)

        # Save model and checkpoint
        torch.save(model.state_dict(), os.path.join(args.log_dir, 'net.pth'))
        if epoch % args.save_every_epochs == 0:
            torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.log_dir, 'ckpt_e{}.pth'.format(epoch + 1)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFDNet")
    parser.add_argument("--gray", action='store_true', help='Train with grayscale images.')
    parser.add_argument("--log_dir", type=str, default="logs", help='Path of log files')
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")  # Reduced batch size
    parser.add_argument("--epochs", "--e", type=int, default=80, help="Number of total training epochs")
    parser.add_argument("--resume_training", "--r", action='store_true', help="Resume training from a previous checkpoint")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--save_every", type=int, default=10, help="Number of training steps to log PSNR")
    parser.add_argument("--save_every_epochs", type=int, default=5, help="Number of training epochs to save state")
    parser.add_argument("--val_noiseL", type=float, default=25, help='Noise level used on validation set')

    argspar = parser.parse_args()

    (train_data, val_data) = load_data(r"G:\SIDD\GT", r"G:\SIDD\NOISY")
    main(argspar, train_data, val_data)    