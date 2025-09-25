import os
import numpy as np
import torch
from PIL import Image
import random
import argparse
import numpy as np


def unpickle(file):
    """Unpickle the CIFAR data files"""
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

def split_cifar100(data_path, label_file, output_dir, idn_ratio, split_ratio=0.8, seed=42):
    """
    Split CIFAR-100 training data into train and calibration sets,
    preserving all annotator labels from the .pt file.
    
    Args:
        data_path: Path to the CIFAR-100 data directory
        label_file: Path to the .pt file containing various annotator labels
        split_ratio: Ratio of data to use for training (default: 0.9)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing the split datasets and labels
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load CIFAR-100 training data
    train_dict = unpickle(f'{data_path}/train')
    train_data = train_dict['data']
    train_data = train_data.reshape((50000, 3, 32, 32))
    train_data = train_data.transpose((0, 2, 3, 1))  # Convert to HWC format
    
    # Original noisy labels from CIFAR-100
    train_original_labels = train_dict['fine_labels']
    
    # Load all annotator labels from the .pt file
    multi_rater = torch.load(os.path.join(data_path, label_file))
    print(f"Available annotators in label file: {list(multi_rater.keys())}")
    
    # Create indices for all samples
    indices = list(range(len(train_data)))
    
    # Shuffle indices
    random.shuffle(indices)
    
    # Calculate split point
    split_point = int(len(indices) * split_ratio)
    
    # Create train and calibration indices
    train_indices = indices[:split_point]
    calibration_indices = indices[split_point:]

    train_labels = {key: np.array([values[i] for i in train_indices]) for key, values in multi_rater.items()}
    train_labels['indices'] = train_indices

    calibration_labels = {key: np.array([values[i] for i in calibration_indices]) for key, values in multi_rater.items()}
    calibration_labels['indices'] = calibration_indices

    train_file = os.path.join(output_dir, f'cifar100_split_train_noise_{idn_ratio}_{split_ratio}.pt')
    calibration_file = os.path.join(output_dir, f'cifar100_split_calibration_noise_{idn_ratio}_{split_ratio}.pt')

    torch.save(train_labels, train_file)
    torch.save(calibration_labels, calibration_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split CIFAR-100 into train and calibration sets')
    parser.add_argument('--data_path', type=str, default='./cifar-100-python', 
                        help='Path to the CIFAR-100 data directory')
    parser.add_argument('--label_file', type=str, required=True,
                        help='Path to the .pt file containing annotator labels')
    parser.add_argument('--output_dir', type=str, default='./cifar100_split',
                        help='Directory to save the split data')
    parser.add_argument('--split_ratio', type=float, default=0.9,
                        help='Ratio of data to use for training (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--idn_noise_rate', type=int, default=50,
                        help='Noise rate for IDN (default: 50)')
    
    args = parser.parse_args()

    split_cifar100(args.data_path, args.label_file, args.output_dir, args.idn_noise_rate, args.split_ratio, args.seed)
    print(f"Data split completed. Train and calibration sets saved to {args.output_dir}.")
    