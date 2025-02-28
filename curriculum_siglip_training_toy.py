#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for comparing curriculum learning vs supervised learning
with SigLIP loss on COCO dataset.


python curriculum_siglip_training_toy.py --dataset-size 1000 --batch-size 16 --epochs-per-stage 3 --resolutions 112 168 224

python curriculum_siglip_training_toy.py --dataset-size 100 --batch-size 8 --epochs-per-stage 1 --resolutions 64 128 --final-resolution 128

python curriculum_siglip_training_toy.py --dataset-size 500 --test-images path/to/image1.jpg path/to/image2.jpg
"""

import os
import json
import argparse
import logging
from datetime import datetime
import numpy as np
from scipy.integrate import trapezoid as trapz

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from PIL import Image

# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train CLIP model with SigLIP loss using curriculum learning')
    parser.add_argument('--dataset', type=str, default='flickr8k',
                        choices=['flickr8k', 'conceptual_captions', 'cifar10', 'cifar100', 'custom', 'coco'],
                        help='Dataset to use for training')
    parser.add_argument('--custom-data-dir', type=str, default=None,
                        help='Directory containing custom dataset (images and captions.txt)')
    parser.add_argument('--dataset-size', type=int, default=1000, help='Number of samples to use from COCO dataset')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epochs-per-stage', type=int, default=3, help='Number of epochs per resolution stage')
    parser.add_argument('--resolutions', type=int, nargs='+', default=[112, 168, 224], help='Resolutions for curriculum learning')
    parser.add_argument('--final-resolution', type=int, default=224, help='Final resolution for supervised learning')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save output files')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Directory to save log files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--validation-split', type=float, default=0.1, help='Fraction of data to use for validation')
    parser.add_argument('--checkpoint-interval', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--temp-init', type=float, default=0.0, help='Initial temperature for SigLIP loss')
    parser.add_argument('--bias-init', type=float, default=0.0, help='Initial bias for SigLIP loss')
    parser.add_argument('--early-stopping', type=int, default=3, help='Stop training if no improvement for N epochs (0 to disable)')
    parser.add_argument('--test-images', type=str, nargs='+', default=[], help='Paths to test images')
    return parser.parse_args()

# Set up logging
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    # Configure logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    if logger.handlers:
        logger.handlers = []

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

class SigLIPLoss(nn.Module):
    """
    SigLIP Loss function as described in the paper "Sigmoid Loss for Language Image Pre-Training"
    This implements a learnable temperature and bias parameter.
    """
    def __init__(self, temp_init=0.0, bias_init=0.0):
        super(SigLIPLoss, self).__init__()
        # Define learnable parameters as part of the module
        self.log_temp = nn.Parameter(torch.tensor(temp_init))
        self.bias = nn.Parameter(torch.tensor(bias_init))

    def forward(self, image_embeds, text_embeds):
        """
        Compute SigLIP loss between image and text embeddings

        Args:
            image_embeds: Tensor of shape (batch_size, embed_dim)
            text_embeds: Tensor of shape (batch_size, embed_dim)

        Returns:
            loss: Scalar loss value
            metrics: Dictionary of metrics for monitoring
        """
        try:
            # Normalize embeddings
            image_embeds = F.normalize(image_embeds, p=2, dim=-1)
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)

            # Compute temperature and logits
            temp = torch.exp(self.log_temp)
            logits = torch.matmul(image_embeds, text_embeds.T) * temp + self.bias

            # Create labels: 1 for positive pairs (diagonal), -1 for negative pairs
            n = image_embeds.shape[0]
            labels = 2 * torch.eye(n, device=logits.device) - torch.ones(n, n, device=logits.device)

            # Compute pairwise sigmoid loss
            loss = -torch.mean(F.logsigmoid(labels * logits))

            # Return additional metrics for monitoring
            with torch.no_grad():
                pos_logits = torch.diag(logits)
                neg_logits = logits - torch.diag(pos_logits)
                avg_pos = torch.mean(pos_logits)
                avg_neg = torch.mean(neg_logits)

            return loss, {
                'temp': temp.item(),
                'bias': self.bias.item(),
                'avg_pos_logit': avg_pos.item(),
                'avg_neg_logit': avg_neg.item()
            }

        except Exception as e:
            logging.error(f"Error in SigLIP loss calculation: {str(e)}")
            # Return a default loss that can be backpropagated
            return torch.tensor(10.0, requires_grad=True, device=image_embeds.device), {
                'temp': 0.0,
                'bias': 0.0,
                'avg_pos_logit': 0.0,
                'avg_neg_logit': 0.0
            }


class ImageTextDataset(Dataset):
    """
    Flexible dataset class that adapts to different dataset structures
    """
    def __init__(self, dataset, processor, resolution, dataset_type='flickr8k'):
        self.dataset = dataset
        self.processor = processor
        self.resolution = resolution
        self.dataset_type = dataset_type
        logging.info(f"Initialized {dataset_type} dataset with {len(dataset)} samples at resolution {resolution}x{resolution}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]

            # Handle different dataset structures
            if self.dataset_type == 'flickr8k':
                # Flickr8k has 'image' and 'caption' fields
                image = item["image"].convert("RGB")
                text = item["caption"] if isinstance(item["caption"], str) else item["caption"][0]

            elif self.dataset_type == 'conceptual_captions':
                # Conceptual Captions has 'image' and 'caption' fields
                image = item["image"].convert("RGB")
                text = item["caption"]

            elif self.dataset_type == 'cifar10' or self.dataset_type == 'cifar100':
                # CIFAR has 'img' and 'label' fields
                image = Image.fromarray(item["img"])
                # Generate a simple caption from the label
                label_name = item["fine_label_name"] if "fine_label_name" in item else item["label_name"] if "label_name" in item else f"Class {item['label']}"
                text = f"A photo of a {label_name}"

            elif self.dataset_type == 'coco':
                # COCO dataset
                image = item["image"].convert("RGB")
                if not item["captions"]:
                    text = "An image"
                else:
                    text = item["captions"][0]  # Take first caption

            elif self.dataset_type == 'custom':
                # Custom dataset
                image = item["image"].convert("RGB")
                text = item["caption"]

            else:
                # Default fallback
                image = item["image"].convert("RGB") if hasattr(item["image"], "convert") else Image.fromarray(item["image"])
                text = item.get("caption", item.get("captions", ["An image"])[0])

            # Apply transforms to resize the image
            transform = transforms.Compose([
                transforms.Resize((self.resolution, self.resolution)),
                transforms.ToTensor(),
            ])

            # Transform the image
            image_tensor = transform(image)

            # Return the raw tensors and text for batch processing
            return {
                "image": image_tensor,
                "text": text,
                "idx": idx
            }

        except Exception as e:
            logging.error(f"Error processing item {idx}: {str(e)}")
            # Return a default item
            dummy_image = torch.zeros(3, self.resolution, self.resolution)
            return {
                "image": dummy_image,
                "text": "Error loading image",
                "idx": idx
            }

class CustomDataset(Dataset):
    """
    Dataset for loading custom images and captions from a directory
    """
    def __init__(self, data_dir, resolution):
        self.data_dir = data_dir
        self.resolution = resolution
        self.image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Load captions
        self.captions = {}
        caption_file = os.path.join(data_dir, 'captions.txt')
        if os.path.exists(caption_file):
            with open(caption_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        self.captions[parts[0]] = parts[1]

        logging.info(f"Loaded {len(self.image_files)} images from {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            image_file = self.image_files[idx]
            image_path = os.path.join(self.data_dir, image_file)
            image = Image.open(image_path).convert("RGB")

            # Get caption if available, otherwise use a default
            caption = self.captions.get(image_file, f"An image named {image_file}")

            # Apply transforms
            transform = transforms.Compose([
                transforms.Resize((self.resolution, self.resolution)),
                transforms.ToTensor(),
            ])

            image_tensor = transform(image)

            return {
                "image": image_tensor,
                "caption": caption,
                "idx": idx
            }

        except Exception as e:
            logging.error(f"Error loading image {self.image_files[idx]}: {str(e)}")
            dummy_image = torch.zeros(3, self.resolution, self.resolution)
            return {
                "image": dummy_image,
                "caption": "Error loading image",
                "idx": idx
            }

def prepare_datasets(args, processor):
    """
    Prepare training and validation datasets based on the selected dataset
    """
    logging.info(f"Loading {args.dataset} dataset with {args.dataset_size} samples")
    try:
        if args.dataset == 'flickr8k':
            full_dataset = load_dataset("nlphuji/flickr8k", split=f"train[:{args.dataset_size}]")

        elif args.dataset == 'conceptual_captions':
            full_dataset = load_dataset("conceptual_captions", "small_subset", split=f"train[:{args.dataset_size}]")

        elif args.dataset == 'cifar10':
            full_dataset = load_dataset("cifar10", split=f"train[:{args.dataset_size}]")

        elif args.dataset == 'cifar100':
            full_dataset = load_dataset("cifar100", split=f"train[:{args.dataset_size}]")

        elif args.dataset == 'coco':
            full_dataset = load_dataset("HuggingFaceM4/COCO", split=f"train[:{args.dataset_size}]")

        elif args.dataset == 'custom':
            if args.custom_data_dir is None:
                raise ValueError("Must specify --custom-data-dir when using custom dataset")

            full_dataset = CustomDataset(args.custom_data_dir, args.resolutions[0])

        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        # Calculate split sizes
        val_size = int(len(full_dataset) * args.validation_split)
        train_size = len(full_dataset) - val_size

        # Split dataset
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )

        logging.info(f"Split dataset into {train_size} training and {val_size} validation samples")

        return train_dataset, val_dataset
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise

'''

class COCODataset(Dataset):
    """
    Custom Dataset wrapper for COCO dataset with image resizing and text processing
    """
    def __init__(self, dataset, processor, resolution):
        self.dataset = dataset
        self.processor = processor
        self.resolution = resolution
        logging.info(f"Initialized COCO dataset with {len(dataset)} samples at resolution {resolution}x{resolution}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        try:
            item = self.dataset[idx]

            # Handle different dataset structures
            if self.dataset_type == 'flickr8k':
                # Flickr8k has 'image' and 'caption' fields
                image = item["image"].convert("RGB")
                text = item["caption"] if isinstance(item["caption"], str) else item["caption"][0]

            elif self.dataset_type == 'conceptual_captions':
                # Conceptual Captions has 'image' and 'caption' fields
                image = item["image"].convert("RGB")
                text = item["caption"]

            elif self.dataset_type == 'cifar10' or self.dataset_type == 'cifar100':
                # CIFAR has 'img' and 'label' fields
                image = Image.fromarray(item["img"])
                # Generate a simple caption from the label
                label_name = item["fine_label_name"] if "fine_label_name" in item else item["label_name"] if "label_name" in item else f"Class {item['label']}"
                text = f"A photo of a {label_name}"

            elif self.dataset_type == 'coco':
                # COCO dataset
                image = item["image"].convert("RGB")
                if not item["captions"]:
                    text = "An image"
                else:
                    text = item["captions"][0]  # Take first caption

            elif self.dataset_type == 'custom':
                # Custom dataset
                image = item["image"].convert("RGB")
                text = item["caption"]

            else:
                # Default fallback
                image = item["image"].convert("RGB") if hasattr(item["image"], "convert") else Image.fromarray(item["image"])
                text = item.get("caption", item.get("captions", ["An image"])[0])

            # Apply transforms to resize the image
            transform = transforms.Compose([
                transforms.Resize((self.resolution, self.resolution)),
                transforms.ToTensor(),
            ])

            # Transform the image
            image_tensor = transform(image)

            # Return the raw tensors and text for batch processing
            return {
                "image": image_tensor,
                "text": text,
                "idx": idx
            }

        except Exception as e:
            logging.error(f"Error processing item {idx}: {str(e)}")
            # Return a default item
            dummy_image = torch.zeros(3, self.resolution, self.resolution)
            return {
                "image": dummy_image,
                "text": "Error loading image",
                "idx": idx
            }
        
        try:
            item = self.dataset[idx]
            image = item["image"].convert("RGB")

            # Handle cases where there might be no captions
            if not item["captions"]:
                text = "An image"
            else:
                text = item["captions"][0]  # Take first caption

            # Apply transforms to resize the image
            transform = transforms.Compose([
                transforms.Resize((self.resolution, self.resolution)),
                transforms.ToTensor(),
            ])

            # Transform the image
            image_tensor = transform(image)

            # Return the raw tensors and text for batch processing
            return {
                "image": image_tensor,
                "text": text,
                "idx": idx
            }

        except Exception as e:
            logging.error(f"Error processing item {idx}: {str(e)}")
            # Return a default item
            dummy_image = torch.zeros(3, self.resolution, self.resolution)
            return {
                "image": dummy_image,
                "text": "Error loading image",
                "idx": idx
            }

'''

# def prepare_datasets(args, processor):
#     """
#     Prepare training and validation datasets
#     """
#     logging.info(f"Loading COCO dataset with {args.dataset_size} samples")
#     try:
#         full_dataset = load_dataset("HuggingFaceM4/COCO", split=f"train[:{args.dataset_size}]")

#         # Calculate split sizes
#         val_size = int(len(full_dataset) * args.validation_split)
#         train_size = len(full_dataset) - val_size

#         # Split dataset
#         train_dataset, val_dataset = random_split(
#             full_dataset,
#             [train_size, val_size],
#             generator=torch.Generator().manual_seed(args.seed)
#         )

#         logging.info(f"Split dataset into {train_size} training and {val_size} validation samples")

#         return train_dataset, val_dataset
#     except Exception as e:
#         logging.error(f"Error loading dataset: {str(e)}")
#         raise



def evaluate_model(model, loss_fn, dataloader, device, resolution, processor):
    """
    Evaluate model on validation set
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    metrics = {
        'temp': 0.0,
        'bias': 0.0,
        'avg_pos_logit': 0.0,
        'avg_neg_logit': 0.0
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating at {resolution}x{resolution}"):
            images = batch["image"].to(device)
            texts = batch["text"]

            try:
                # Process inputs through CLIP processor
                inputs = processor(
                    text=texts,
                    images=[img for img in images],  # Convert tensor to list of images
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )

                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Forward pass
                outputs = model(**inputs)

                # Get embeddings
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds

                # Compute loss
                loss, batch_metrics = loss_fn(image_embeds, text_embeds)
                total_loss += loss.item()

                # Update metrics
                for k, v in batch_metrics.items():
                    metrics[k] += v

                # Calculate accuracy (correct if highest similarity is on diagonal)
                similarity = torch.matmul(F.normalize(image_embeds, dim=-1), F.normalize(text_embeds, dim=-1).T)
                predictions = torch.argmax(similarity, dim=1)
                targets = torch.arange(predictions.size(0), device=device)
                correct += (predictions == targets).sum().item()
                total += predictions.size(0)
            except Exception as e:
                logging.error(f"Error during evaluation: {str(e)}")
                continue

    # Calculate average metrics
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    accuracy = correct / total if total > 0 else 0

    for k in metrics:
        metrics[k] /= len(dataloader) if len(dataloader) > 0 else 1

    return avg_loss, accuracy, metrics

def train_model(model, loss_fn, train_loader, val_loader, optimizer, args, device, 
                resolution, stage_name, history, checkpoint_dir, processor):
    """
    Train model for a specific resolution stage
    """
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs_per_stage):
        # Training phase
        model.train()
        running_loss = 0.0
        train_metrics = {
            'temp': 0.0,
            'bias': 0.0,
            'avg_pos_logit': 0.0,
            'avg_neg_logit': 0.0
        }
        
        # Use tqdm for progress bar
        train_iter = tqdm(train_loader, desc=f"{stage_name} Epoch [{epoch+1}/{args.epochs_per_stage}]")
        for i, batch in enumerate(train_iter):
            try:
                images = batch["image"].to(device)
                texts = batch["text"]
                
                # Process inputs through CLIP processor
                inputs = processor(
                    text=texts,
                    images=[img for img in images],  # Convert tensor to list of images
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = model(**inputs)
                
                # Get embeddings
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                
                # Compute loss
                optimizer.zero_grad()
                loss, batch_metrics = loss_fn(image_embeds, text_embeds)
                loss.backward()
                optimizer.step()
                
                # Update metrics
                running_loss += loss.item()
                for k, v in batch_metrics.items():
                    train_metrics[k] += v
                    
                # Update progress bar
                if (i + 1) % 10 == 0:
                    train_iter.set_postfix({
                        'loss': running_loss / (i + 1),
                        'temp': batch_metrics['temp'],
                        'bias': batch_metrics['bias']
                    })
            except Exception as e:
                logging.error(f"Error during training batch {i}: {str(e)}")
                continue
        
        # Calculate average training metrics
        avg_train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        for k in train_metrics:
            train_metrics[k] /= len(train_loader) if len(train_loader) > 0 else 1
        
        # Validation phase
        val_loss, val_accuracy, val_metrics = evaluate_model(
            model, loss_fn, val_loader, device, resolution, processor
        )
        
        # Log results
        logging.info(f"{stage_name} Epoch [{epoch+1}/{args.epochs_per_stage}], "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy:.4f}, "
                    f"Temp: {val_metrics['temp']:.4f}, "
                    f"Bias: {val_metrics['bias']:.4f}")
        
        # Save history
        history['loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(val_accuracy)
        history['temp'].append(val_metrics['temp'])
        history['bias'].append(val_metrics['bias'])
        history['resolution'].append(resolution)
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"{stage_name.lower().replace(' ', '_')}_res{resolution}_checkpoint_epoch_{epoch+1}.pth.tar"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_fn_state_dict': loss_fn.state_dict(),
                'loss': avg_train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'resolution': resolution,
            }, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, f"{stage_name.lower().replace(' ', '_')}_model_best.pth.tar")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_fn_state_dict': loss_fn.state_dict(),
                'loss': avg_train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'resolution': resolution,
            }, best_model_path)
            logging.info(f"Best model saved to {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save history to CSV
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(checkpoint_dir, f"{stage_name.lower().replace(' ', '_')}_history.csv"), index=False)
        stage_name_rep = stage_name.lower().replace(' ', '_')
        path = os.path.join(checkpoint_dir, f'{stage_name_rep}')
        logging.info(f"Training history saved to {path}_history.csv')")
    
    return history, best_val_loss

def plot_training_curves(curriculum_history, supervised_history, save_dir):
    """
    Create interactive plots comparing curriculum and supervised learning
    """
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Convert to DataFrames for easier handling
        curriculum_df = pd.DataFrame(curriculum_history)
        supervised_df = pd.DataFrame(supervised_history)

        # Add epoch column
        curriculum_df['epoch'] = range(1, len(curriculum_df) + 1)
        supervised_df['epoch'] = range(1, len(supervised_df) + 1)

        # Save raw data
        curriculum_df.to_csv(os.path.join(save_dir, 'curriculum_history.csv'), index=False)
        supervised_df.to_csv(os.path.join(save_dir, 'supervised_history.csv'), index=False)
        logging.info(f"Training data saved to {save_dir}")

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Validation Accuracy', 'Temperature', 'Bias'),
            shared_xaxes=True
        )

        # Add traces for loss
        fig.add_trace(
            go.Scatter(
                x=curriculum_df['epoch'],
                y=curriculum_df['loss'],
                mode='lines+markers',
                name='Curriculum Loss',
                line=dict(color='blue'),
                customdata=curriculum_df['resolution'],
                hovertemplate='Epoch: %{x}<br>Loss: %{y:.4f}<br>Resolution: %{customdata}x%{customdata}'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=supervised_df['epoch'],
                y=supervised_df['loss'],
                mode='lines+markers',
                name='Supervised Loss',
                line=dict(color='red'),
                hovertemplate='Epoch: %{x}<br>Loss: %{y:.4f}'
            ),
            row=1, col=1
        )

        # Add traces for accuracy
        fig.add_trace(
            go.Scatter(
                x=curriculum_df['epoch'],
                y=curriculum_df['accuracy'],
                mode='lines+markers',
                name='Curriculum Accuracy',
                line=dict(color='blue'),
                customdata=curriculum_df['resolution'],
                hovertemplate='Epoch: %{x}<br>Accuracy: %{y:.4f}<br>Resolution: %{customdata}x%{customdata}'
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=supervised_df['epoch'],
                y=supervised_df['accuracy'],
                mode='lines+markers',
                name='Supervised Accuracy',
                line=dict(color='red'),
                hovertemplate='Epoch: %{x}<br>Accuracy: %{y:.4f}'
            ),
            row=1, col=2
        )

        # Add traces for temperature
        fig.add_trace(
            go.Scatter(
                x=curriculum_df['epoch'],
                y=curriculum_df['temp'],
                mode='lines+markers',
                name='Curriculum Temp',
                line=dict(color='blue'),
                customdata=curriculum_df['resolution'],
                hovertemplate='Epoch: %{x}<br>Temp: %{y:.4f}<br>Resolution: %{customdata}x%{customdata}'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=supervised_df['epoch'],
                y=supervised_df['temp'],
                mode='lines+markers',
                name='Supervised Temp',
                line=dict(color='red'),
                hovertemplate='Epoch: %{x}<br>Temp: %{y:.4f}'
            ),
            row=2, col=1
        )

        # Add traces for bias
        fig.add_trace(
            go.Scatter(
                x=curriculum_df['epoch'],
                y=curriculum_df['bias'],
                mode='lines+markers',
                name='Curriculum Bias',
                line=dict(color='blue'),
                customdata=curriculum_df['resolution'],
                hovertemplate='Epoch: %{x}<br>Bias: %{y:.4f}<br>Resolution: %{customdata}x%{customdata}'
            ),
            row=2, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=supervised_df['epoch'],
                y=supervised_df['bias'],
                mode='lines+markers',
                name='Supervised Bias',
                line=dict(color='red'),
                hovertemplate='Epoch: %{x}<br>Bias: %{y:.4f}'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text="Training Comparison: Curriculum vs Supervised Learning",
            hovermode="closest"
        )

        # Add range slider
        fig.update_xaxes(rangeslider_visible=True, row=2, col=1)
        fig.update_xaxes(rangeslider_visible=True, row=2, col=2)

        # Save as interactive HTML
        html_path = os.path.join(save_dir, f'interactive_training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
        fig.write_html(html_path)
        logging.info(f"Interactive training curves saved to {html_path}")

        # Save as static image
        try:
            png_path = os.path.join(save_dir, f'training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            fig.write_image(png_path)
            logging.info(f"Static training curves saved to {png_path}")
        except Exception as img_error:
            logging.error(f"Error saving static image: {str(img_error)}")

        # Save as JSON for later loading
        json_path = os.path.join(save_dir, f'training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        fig.write_json(json_path)
        logging.info(f"Training curves data saved to {json_path}")

    except Exception as e:
        logging.error(f"Error plotting with Plotly: {str(e)}")

        # Fallback to matplotlib
        try:
            logging.info("Falling back to matplotlib for static plots")
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 10))

            # Plot loss
            plt.subplot(2, 2, 1)
            plt.plot(curriculum_df['epoch'], curriculum_df['loss'], 'b-', label='Curriculum Loss')
            plt.plot(supervised_df['epoch'], supervised_df['loss'], 'r-', label='Supervised Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()

            # Plot accuracy
            plt.subplot(2, 2, 2)
            plt.plot(curriculum_df['epoch'], curriculum_df['accuracy'], 'b-', label='Curriculum Accuracy')
            plt.plot(supervised_df['epoch'], supervised_df['accuracy'], 'r-', label='Supervised Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Validation Accuracy')
            plt.legend()

            # Plot temperature
            plt.subplot(2, 2, 3)
            plt.plot(curriculum_df['epoch'], curriculum_df['temp'], 'b-', label='Curriculum Temp')
            plt.plot(supervised_df['epoch'], supervised_df['temp'], 'r-', label='Supervised Temp')
            plt.xlabel('Epoch')
            plt.ylabel('Temperature')
            plt.title('Temperature Parameter')
            plt.legend()

            # Plot bias
            plt.subplot(2, 2, 4)
            plt.plot(curriculum_df['epoch'], curriculum_df['bias'], 'b-', label='Curriculum Bias')
            plt.plot(supervised_df['epoch'], supervised_df['bias'], 'r-', label='Supervised Bias')
            plt.xlabel('Epoch')
            plt.ylabel('Bias')
            plt.title('Bias Parameter')
            plt.legend()

            plt.tight_layout()

            # Save figure
            fallback_path = os.path.join(save_dir, f'training_curves_fallback_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(fallback_path)
            plt.close()
            logging.info(f"Fallback training curves saved to {fallback_path}")

        except Exception as e2:
            logging.error(f"Error with fallback plotting: {str(e2)}")

def calculate_metrics(curriculum_history, supervised_history):
    """
    Calculate metrics for comparing curriculum and supervised learning
    """
    metrics = {}
    
    try:
        # Convert to numpy arrays for calculations
        curriculum_acc = np.array(curriculum_history['accuracy'])
        supervised_acc = np.array(supervised_history['accuracy'])
        
        # Calculate AUC for accuracy
        curriculum_epochs = np.arange(1, len(curriculum_acc) + 1)
        supervised_epochs = np.arange(1, len(supervised_acc) + 1)
        
        curriculum_auc = trapz(curriculum_acc, x=curriculum_epochs)
        supervised_auc = trapz(supervised_acc, x=supervised_epochs)
        
        # Normalize by number of epochs
        curriculum_auc_normalized = curriculum_auc / len(curriculum_epochs)
        supervised_auc_normalized = supervised_auc / len(supervised_epochs)

        # Calculate final accuracy
        curriculum_final_acc = curriculum_acc[-1] if len(curriculum_acc) > 0 else 0
        supervised_final_acc = supervised_acc[-1] if len(supervised_acc) > 0 else 0

        # Calculate best accuracy
        curriculum_best_acc = np.max(curriculum_acc) if len(curriculum_acc) > 0 else 0
        supervised_best_acc = np.max(supervised_acc) if len(supervised_acc) > 0 else 0

        # Calculate convergence speed (epochs to reach 90% of best accuracy)
        threshold_curriculum = 0.9 * curriculum_best_acc
        threshold_supervised = 0.9 * supervised_best_acc

        convergence_curriculum = np.argmax(curriculum_acc >= threshold_curriculum) + 1 if np.any(curriculum_acc >= threshold_curriculum) else len(curriculum_acc)
        convergence_supervised = np.argmax(supervised_acc >= threshold_supervised) + 1 if np.any(supervised_acc >= threshold_supervised) else len(supervised_acc)

        # Store metrics
        metrics['curriculum_auc'] = float(curriculum_auc)
        metrics['supervised_auc'] = float(supervised_auc)
        metrics['curriculum_auc_normalized'] = float(curriculum_auc_normalized)
        metrics['supervised_auc_normalized'] = float(supervised_auc_normalized)
        metrics['curriculum_final_acc'] = float(curriculum_final_acc)
        metrics['supervised_final_acc'] = float(supervised_final_acc)
        metrics['curriculum_best_acc'] = float(curriculum_best_acc)
        metrics['supervised_best_acc'] = float(supervised_best_acc)
        metrics['curriculum_convergence'] = int(convergence_curriculum)
        metrics['supervised_convergence'] = int(convergence_supervised)

        # Calculate late-stage performance (last 1/3 of training)
        if len(curriculum_acc) >= 3:
            late_start = len(curriculum_acc) // 3 * 2
            curriculum_late_acc = curriculum_acc[late_start:]
            curriculum_late_epochs = curriculum_epochs[late_start:]
            curriculum_late_auc = trapz(curriculum_late_acc, x=curriculum_late_epochs)
            curriculum_late_auc_normalized = curriculum_late_auc / len(curriculum_late_epochs)
            metrics['curriculum_late_auc'] = float(curriculum_late_auc)
            metrics['curriculum_late_auc_normalized'] = float(curriculum_late_auc_normalized)

        if len(supervised_acc) >= 3:
            late_start = len(supervised_acc) // 3 * 2
            supervised_late_acc = supervised_acc[late_start:]
            supervised_late_epochs = supervised_epochs[late_start:]
            supervised_late_auc = trapz(supervised_late_acc, x=supervised_late_epochs)
            supervised_late_auc_normalized = supervised_late_auc / len(supervised_late_epochs)
            metrics['supervised_late_auc'] = float(supervised_late_auc)
            metrics['supervised_late_auc_normalized'] = float(supervised_late_auc_normalized)

        # Log metrics
        logging.info(f"Curriculum Learning AUC: {curriculum_auc:.4f} (normalized: {curriculum_auc_normalized:.4f})")
        logging.info(f"Supervised Learning AUC: {supervised_auc:.4f} (normalized: {supervised_auc_normalized:.4f})")
        logging.info(f"Curriculum Learning Final Accuracy: {curriculum_final_acc:.4f}")
        logging.info(f"Supervised Learning Final Accuracy: {supervised_final_acc:.4f}")
        logging.info(f"Curriculum Learning Best Accuracy: {curriculum_best_acc:.4f}")
        logging.info(f"Supervised Learning Best Accuracy: {supervised_best_acc:.4f}")
        logging.info(f"Curriculum Learning Convergence: {convergence_curriculum} epochs")
        logging.info(f"Supervised Learning Convergence: {convergence_supervised} epochs")

        # Calculate improvement percentages
        if supervised_auc_normalized > 0:
            auc_improvement = (curriculum_auc_normalized - supervised_auc_normalized) / supervised_auc_normalized * 100
            metrics['auc_improvement_percent'] = float(auc_improvement)
            logging.info(f"AUC Improvement: {auc_improvement:.2f}%")

        if supervised_final_acc > 0:
            final_acc_improvement = (curriculum_final_acc - supervised_final_acc) / supervised_final_acc * 100
            metrics['final_acc_improvement_percent'] = float(final_acc_improvement)
            logging.info(f"Final Accuracy Improvement: {final_acc_improvement:.2f}%")

        if supervised_convergence > 0:
            convergence_improvement = (supervised_convergence - convergence_curriculum) / supervised_convergence * 100
            metrics['convergence_improvement_percent'] = float(convergence_improvement)
            logging.info(f"Convergence Speed Improvement: {convergence_improvement:.2f}%")

    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        # Set default metrics
        metrics = {
            'curriculum_auc': 0.0,
            'supervised_auc': 0.0,
            'curriculum_auc_normalized': 0.0,
            'supervised_auc_normalized': 0.0,
            'curriculum_final_acc': 0.0,
            'supervised_final_acc': 0.0,
            'curriculum_best_acc': 0.0,
            'supervised_best_acc': 0.0,
            'curriculum_convergence': 0,
            'supervised_convergence': 0,
            'error': str(e)
        }

    return metrics

def first_epoch_above_threshold(history, threshold):
    """
    Find the first epoch where accuracy exceeds the threshold
    """
    for i, acc in enumerate(history['accuracy']):
        if acc >= threshold:
            return i + 1  # Epochs are 1-indexed
    return len(history['accuracy']) + 1  # Return one more than total epochs if threshold never reached

def run_supervised_training(args, train_dataset, val_dataset, processor, device):
    """
    Run supervised training with fixed resolution
    """
    logging.info(f"Starting supervised training with resolution {args.final_resolution}x{args.final_resolution}")

    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.checkpoint_dir, 'supervised')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize model and loss function
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    loss_fn = SigLIPLoss(temp_init=args.temp_init, bias_init=args.bias_init)

    # Move to device
    model.to(device)
    loss_fn.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=args.learning_rate
    )

    # Create datasets
    train_coco = COCODataset(train_dataset, processor, args.final_resolution)
    val_coco = COCODataset(val_dataset, processor, args.final_resolution)

    # Create dataloaders
    train_loader = DataLoader(
        train_coco,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_coco,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Initialize history
    history = {
        'loss': [],
        'val_loss': [],
        'accuracy': [],
        'temp': [],
        'bias': [],
        'resolution': []
    }

    # Train for all epochs
    total_epochs = args.epochs_per_stage * len(args.resolutions)  # Match curriculum total epochs

    # Train model
    for epoch_group in range(len(args.resolutions)):
        logging.info(f"Starting epoch group {epoch_group+1}/{len(args.resolutions)}")
        history, best_val_loss = train_model(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            args=args,
            device=device,
            resolution=args.final_resolution,
            stage_name=f"Supervised Group {epoch_group+1}",
            history=history,
            checkpoint_dir=checkpoint_dir,
            processor=processor  # Pass processor
        )

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "supervised_final_model.pth.tar")
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss_fn_state_dict': loss_fn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, final_model_path)
    logging.info(f"Final supervised model saved to {final_model_path}")

    return model, loss_fn, history

def run_curriculum_training(args, train_dataset, val_dataset, processor, device):
    """
    Run curriculum training with increasing resolutions
    """
    logging.info(f"Starting curriculum training with resolutions {args.resolutions}")

    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.checkpoint_dir, 'curriculum')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize model and loss function
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    loss_fn = SigLIPLoss(temp_init=args.temp_init, bias_init=args.bias_init)

    # Move to device
    model.to(device)
    loss_fn.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=args.learning_rate
    )

    # Initialize history
    history = {
        'loss': [],
        'val_loss': [],
        'accuracy': [],
        'temp': [],
        'bias': [],
        'resolution': []
    }

    # Train for each resolution
    for i, resolution in enumerate(args.resolutions):
        logging.info(f"Starting curriculum stage {i+1}/{len(args.resolutions)} with resolution {resolution}x{resolution}")

        # Create datasets for current resolution
        train_coco = COCODataset(train_dataset, processor, resolution)
        val_coco = COCODataset(val_dataset, processor, resolution)

        # Create dataloaders
        train_loader = DataLoader(
            train_coco,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )

        val_loader = DataLoader(
            val_coco,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )

        # Train model for this resolution
        history, best_val_loss = train_model(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            args=args,
            device=device,
            resolution=resolution,
            stage_name=f"Curriculum Stage {i+1}",
            history=history,
            checkpoint_dir=checkpoint_dir,
            processor=processor  # Pass processor
        )

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "curriculum_final_model.pth.tar")
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss_fn_state_dict': loss_fn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, final_model_path)
    logging.info(f"Final curriculum model saved to {final_model_path}")

    return model, loss_fn, history

def create_summary_report(args, metrics, curriculum_history, supervised_history):
    """
    Create a summary report of the experiment
    """
    report_path = os.path.join(args.output_dir, f'experiment_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md')

    with open(report_path, 'w') as f:
        f.write("# SigLIP Training Experiment Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Experiment Configuration\n\n")
        f.write(f"- Dataset: COCO (size: {args.dataset_size})\n")
        f.write(f"- Batch Size: {args.batch_size}\n")
        f.write(f"- Learning Rate: {args.learning_rate}\n")
        f.write(f"- Epochs per Stage: {args.epochs_per_stage}\n")
        f.write(f"- Curriculum Resolutions: {args.resolutions}\n")
        f.write(f"- Supervised Resolution: {args.final_resolution}\n")
        f.write(f"- Random Seed: {args.seed}\n")
        f.write(f"- Device: {'CUDA' if torch.cuda.is_available() and not args.no_cuda else 'CPU'}\n\n")

        f.write("## Performance Metrics\n\n")
        f.write("### Accuracy\n\n")
        f.write(f"- Curriculum Learning Final Accuracy: {metrics['curriculum_final_acc']:.4f}\n")
        f.write(f"- Supervised Learning Final Accuracy: {metrics['supervised_final_acc']:.4f}\n")
        if 'final_acc_improvement_percent' in metrics:
            f.write(f"- Improvement: {metrics['final_acc_improvement_percent']:.2f}%\n\n")
        else:
            f.write("\n")

        f.write("### Area Under the Curve (AUC)\n\n")
        f.write(f"- Curriculum Learning AUC: {metrics['curriculum_auc']:.4f} (normalized: {metrics['curriculum_auc_normalized']:.4f})\n")
        f.write(f"- Supervised Learning AUC: {metrics['supervised_auc']:.4f} (normalized: {metrics['supervised_auc_normalized']:.4f})\n")
        if 'auc_improvement_percent' in metrics:
            f.write(f"- Improvement: {metrics['auc_improvement_percent']:.2f}%\n\n")
        else:
            f.write("\n")

        f.write("### Convergence Speed\n\n")
        f.write(f"- Curriculum Learning Convergence: {metrics['curriculum_convergence']} epochs\n")
        f.write(f"- Supervised Learning Convergence: {metrics['supervised_convergence']} epochs\n")
        if 'convergence_improvement_percent' in metrics:
            f.write(f"- Improvement: {metrics['convergence_improvement_percent']:.2f}%\n\n")
        else:
            f.write("\n")

        f.write("### Best Performance\n\n")
        f.write(f"- Curriculum Learning Best Accuracy: {metrics['curriculum_best_acc']:.4f}\n")
        f.write(f"- Supervised Learning Best Accuracy: {metrics['supervised_best_acc']:.4f}\n\n")

        if 'curriculum_late_auc' in metrics and 'supervised_late_auc' in metrics:
            f.write("### Late-Stage Performance\n\n")
            f.write(f"- Curriculum Learning Late-Stage AUC: {metrics['curriculum_late_auc']:.4f} (normalized: {metrics['curriculum_late_auc_normalized']:.4f})\n")
            f.write(f"- Supervised Learning Late-Stage AUC: {metrics['supervised_late_auc']:.4f} (normalized: {metrics['supervised_late_auc_normalized']:.4f})\n\n")

        if 'curriculum_first_epoch_above_threshold' in metrics and 'supervised_first_epoch_above_threshold' in metrics:
            f.write("### Time to Reach Accuracy Threshold\n\n")
            f.write(f"- Threshold: {metrics['accuracy_threshold']:.2f}\n")
            f.write(f"- Curriculum Learning: {metrics['curriculum_first_epoch_above_threshold']} epochs\n")
            f.write(f"- Supervised Learning: {metrics['supervised_first_epoch_above_threshold']} epochs\n\n")

        f.write("## Conclusion\n\n")

        # Determine which method performed better
        if metrics['curriculum_final_acc'] > metrics['supervised_final_acc']:
            f.write("Curriculum learning achieved **higher final accuracy** than supervised learning. ")
        else:
            f.write("Supervised learning achieved **higher final accuracy** than curriculum learning. ")

        if metrics['curriculum_auc_normalized'] > metrics['supervised_auc_normalized']:
            f.write("Curriculum learning had **better overall performance** (higher AUC) throughout training. ")
        else:
            f.write("Supervised learning had **better overall performance** (higher AUC) throughout training. ")

        if metrics['curriculum_convergence'] < metrics['supervised_convergence']:
            f.write("Curriculum learning **converged faster** to a good solution.\n\n")
        else:
            f.write("Supervised learning **converged faster** to a good solution.\n\n")

        f.write("### Training Visualization\n\n")
        f.write("Please refer to the interactive HTML visualization for detailed training curves.\n")

    logging.info(f"Summary report saved to {report_path}")

def test_models_on_images(curriculum_model, supervised_model, processor, test_images, device, output_dir):
    """
    Test both trained models on a set of test images

    Args:
        curriculum_model: Trained curriculum model
        supervised_model: Trained supervised model
        processor: CLIP processor
        test_images: List of image paths or PIL images
        device: Device to run inference on
        output_dir: Directory to save results

    Returns:
        Dictionary of results
    """
    logging.info(f"Testing models on {len(test_images)} images")

    # Ensure models are in eval mode
    curriculum_model.eval()
    supervised_model.eval()

    # Prepare test prompts
    test_prompts = [
        "a photo of a dog",
        "a photo of a cat",
        "a landscape image",
        "a portrait of a person",
        "a picture of food"
    ]

    results = {
        "curriculum": [],
        "supervised": [],
        "comparison": []
    }

    # Process each image
    for i, image_path in enumerate(test_images):
        try:
            # Load image if path is provided
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path

            # Process image and text with both models
            with torch.no_grad():
                # Curriculum model
                inputs = processor(
                    text=test_prompts,
                    images=[image] * len(test_prompts),
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                curriculum_outputs = curriculum_model(**inputs)
                curriculum_logits = torch.matmul(
                    F.normalize(curriculum_outputs.image_embeds, dim=-1),
                    F.normalize(curriculum_outputs.text_embeds, dim=-1).T
                )
                curriculum_probs = F.softmax(curriculum_logits, dim=-1)
                curriculum_scores = curriculum_probs.cpu().numpy()

                # Supervised model
                supervised_outputs = supervised_model(**inputs)
                supervised_logits = torch.matmul(
                    F.normalize(supervised_outputs.image_embeds, dim=-1),
                    F.normalize(supervised_outputs.text_embeds, dim=-1).T
                )
                supervised_probs = F.softmax(supervised_logits, dim=-1)
                supervised_scores = supervised_probs.cpu().numpy()

                # Get top prediction for each model
                curriculum_top_idx = curriculum_scores.argmax(axis=1)
                supervised_top_idx = supervised_scores.argmax(axis=1)

                # Store results
                image_results = {
                    "image_idx": i,
                    "image_path": image_path if isinstance(image_path, str) else f"image_{i}",
                    "curriculum_predictions": [
                        {
                            "prompt": test_prompts[j],
                            "score": float(curriculum_scores[0, j]),
                            "rank": int(np.where(np.argsort(-curriculum_scores[0]))[0][j])
                        }
                        for j in range(len(test_prompts))
                    ],
                    "supervised_predictions": [
                        {
                            "prompt": test_prompts[j],
                            "score": float(supervised_scores[0, j]),
                            "rank": int(np.where(np.argsort(-supervised_scores[0]))[0][j])
                        }
                        for j in range(len(test_prompts))
                    ],
                    "curriculum_top_prompt": test_prompts[curriculum_top_idx[0]],
                    "supervised_top_prompt": test_prompts[supervised_top_idx[0]],
                    "agreement": test_prompts[curriculum_top_idx[0]] == test_prompts[supervised_top_idx[0]]
                }

                results["curriculum"].append(image_results["curriculum_predictions"])
                results["supervised"].append(image_results["supervised_predictions"])
                results["comparison"].append({
                    "image_idx": i,
                    "image_path": image_path if isinstance(image_path, str) else f"image_{i}",
                    "curriculum_top": image_results["curriculum_top_prompt"],
                    "supervised_top": image_results["supervised_top_prompt"],
                    "agreement": image_results["agreement"]
                })

                logging.info(f"Image {i}: Curriculum top: {image_results['curriculum_top_prompt']}, "
                           f"Supervised top: {image_results['supervised_top_prompt']}, "
                           f"Agreement: {image_results['agreement']}")

        except Exception as e:
            logging.error(f"Error processing test image {i}: {str(e)}")
            results["errors"] = results.get("errors", []) + [{"image_idx": i, "error": str(e)}]

    # Calculate agreement rate
    agreement_count = sum(1 for item in results["comparison"] if item["agreement"])
    agreement_rate = agreement_count / len(test_images) if test_images else 0
    results["agreement_rate"] = agreement_rate
    logging.info(f"Model agreement rate: {agreement_rate:.2f} ({agreement_count}/{len(test_images)})")

    # Save results
    results_path = os.path.join(output_dir, f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Test results saved to {results_path}")

    return results

def run_experiment(args):
    """
    Run the complete experiment comparing supervised and curriculum learning
    """
    # Set up logging
    logger = setup_logging(args.log_dir)

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # Load processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(args, processor)

    # Run supervised training
    supervised_model, supervised_loss_fn, supervised_history = run_supervised_training(
        args, train_dataset, val_dataset, processor, device
    )

    # Run curriculum training
    curriculum_model, curriculum_loss_fn, curriculum_history = run_curriculum_training(
        args, train_dataset, val_dataset, processor, device
    )

    # Plot training curves
    plot_training_curves(curriculum_history, supervised_history, args.output_dir)

    # Calculate comparison metrics
    metrics = calculate_metrics(curriculum_history, supervised_history)

    # Find first epoch where accuracy exceeds threshold
    threshold = 0.7  # 70% accuracy threshold
    curriculum_first = first_epoch_above_threshold(curriculum_history, threshold)
    supervised_first = first_epoch_above_threshold(supervised_history, threshold)

    metrics['curriculum_first_epoch_above_threshold'] = curriculum_first
    metrics['supervised_first_epoch_above_threshold'] = supervised_first
    metrics['accuracy_threshold'] = threshold

    logging.info(f"Curriculum Learning first epoch above {threshold}: {curriculum_first}")
    logging.info(f"Supervised Learning first epoch above {threshold}: {supervised_first}")

    # Save metrics to JSON
    metrics_path = os.path.join(args.output_dir, f'comparison_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Comparison metrics saved to {metrics_path}")

    # Create summary report
    create_summary_report(args, metrics, curriculum_history, supervised_history)

    # Test models on images if provided
    if args.test_images:
        test_results = test_models_on_images(
            curriculum_model, supervised_model, processor, args.test_images, device, args.output_dir
        )

    return metrics

def main():
    """
    Main function to run the experiment
    """
    # Parse arguments
    args = parse_args()

    try:
        # Run experiment
        metrics = run_experiment(args)

        # Print final results
        print("\n" + "="*50)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Curriculum Learning Final Accuracy: {metrics['curriculum_final_acc']:.4f}")
        print(f"Supervised Learning Final Accuracy: {metrics['supervised_final_acc']:.4f}")

        if metrics['curriculum_final_acc'] > metrics['supervised_final_acc']:
            print("\nCurriculum learning performed better!")
        else:
            print("\nSupervised learning performed better!")

        print("\nCheck the output directory for detailed results and visualizations.")
        print("="*50)

    except Exception as e:
        logging.error(f"Error in experiment: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        print("\n" + "="*50)
        print("EXPERIMENT FAILED")
        print(f"Error: {str(e)}")
        print("Check logs for details.")
        print("="*50)

# if __name__ == "__main__":
#     main()

def evaluate_model(model, loss_fn, dataloader, device, resolution, processor, dataset_type):
    """
    Evaluate model on validation set
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    metrics = {
        'temp': 0.0,
        'bias': 0.0,
        'avg_pos_logit': 0.0,
        'avg_neg_logit': 0.0
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating at {resolution}x{resolution}"):
            images = batch["image"].to(device)
            texts = batch["text"]

            try:
                # Process inputs through CLIP processor
                inputs = processor(
                    text=texts,
                    images=[img for img in images],  # Convert tensor to list of images
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )

                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Forward pass
                outputs = model(**inputs)

                # Get embeddings
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds

                # Compute loss
                loss, batch_metrics = loss_fn(image_embeds, text_embeds)
                total_loss += loss.item()

                # Update metrics
                for k, v in batch_metrics.items():
                    metrics[k] += v

                # Calculate accuracy (correct if highest similarity is on diagonal)
                similarity = torch.matmul(F.normalize(image_embeds, dim=-1), F.normalize(text_embeds, dim=-1).T)
                predictions = torch.argmax(similarity, dim=1)
                targets = torch.arange(predictions.size(0), device=device)
                correct += (predictions == targets).sum().item()
                total += predictions.size(0)
            except Exception as e:
                logging.error(f"Error during evaluation: {str(e)}")
                continue

    # Calculate average metrics
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    accuracy = correct / total if total > 0 else 0

    for k in metrics:
        metrics[k] /= len(dataloader) if len(dataloader) > 0 else 1

    return avg_loss, accuracy, metrics

def train_model(model, loss_fn, train_loader, val_loader, optimizer, args, device,
                resolution, stage_name, history, checkpoint_dir, processor, dataset_type):
    """
    Train model for a specific resolution stage
    """
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs_per_stage):
        # Training phase
        model.train()
        running_loss = 0.0
        train_metrics = {
            'temp': 0.0,
            'bias': 0.0,
            'avg_pos_logit': 0.0,
            'avg_neg_logit': 0.0
        }

        # Use tqdm for progress bar
        train_iter = tqdm(train_loader, desc=f"{stage_name} Epoch [{epoch+1}/{args.epochs_per_stage}]")
        for i, batch in enumerate(train_iter):
            try:
                images = batch["image"].to(device)
                texts = batch["text"]

                # Process inputs through CLIP processor
                inputs = processor(
                    text=texts,
                    images=[img for img in images],  # Convert tensor to list of images
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )

                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Forward pass
                outputs = model(**inputs)

                # Get embeddings
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds

                # Compute loss
                optimizer.zero_grad()
                loss, batch_metrics = loss_fn(image_embeds, text_embeds)
                loss.backward()
                optimizer.step()

                # Update metrics
                running_loss += loss.item()
                for k, v in batch_metrics.items():
                    train_metrics[k] += v

                # Update progress bar
                if (i + 1) % 10 == 0:
                    train_iter.set_postfix({
                        'loss': running_loss / (i + 1),
                        'temp': batch_metrics['temp'],
                        'bias': batch_metrics['bias']
                    })
            except Exception as e:
                logging.error(f"Error during training batch {i}: {str(e)}")
                continue

        # Calculate average training metrics
        avg_train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        for k in train_metrics:
            train_metrics[k] /= len(train_loader) if len(train_loader) > 0 else 1

        # Validation phase
        val_loss, val_accuracy, val_metrics = evaluate_model(
            model, loss_fn, val_loader, device, resolution, processor, dataset_type
        )

        # Log results
        logging.info(f"{stage_name} Epoch [{epoch+1}/{args.epochs_per_stage}], "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy:.4f}, "
                    f"Temp: {val_metrics['temp']:.4f}, "
                    f"Bias: {val_metrics['bias']:.4f}")

        # Save history
        history['loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(val_accuracy)
        history['temp'].append(val_metrics['temp'])
        history['bias'].append(val_metrics['bias'])
        history['resolution'].append(resolution)

        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"{stage_name.lower().replace(' ', '_')}_res{resolution}_checkpoint_epoch_{epoch+1}.pth.tar"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_fn_state_dict': loss_fn.state_dict(),
                'loss': avg_train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'resolution': resolution,
            }, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, f"{stage_name.lower().replace(' ', '_')}_model_best.pth.tar")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_fn_state_dict': loss_fn.state_dict(),
                'loss': avg_train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'resolution': resolution,
            }, best_model_path)
            logging.info(f"Best model saved to {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Save history to CSV
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(checkpoint_dir, f"{stage_name.lower().replace(' ', '_')}_history.csv"), index=False)
        stage_name_rep = stage_name.lower().replace(' ', '_')
        path = os.path.join(checkpoint_dir, f'{stage_name_rep}')
        logging.info(f"Training history saved to {path}_history.csv')")

    return history, best_val_loss

def run_supervised_training(args, train_dataset, val_dataset, processor, device):
    """
    Run supervised training with fixed resolution
    """
    logging.info(f"Starting supervised training with resolution {args.final_resolution}x{args.final_resolution}")

    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.checkpoint_dir, 'supervised')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize model and loss function
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    loss_fn = SigLIPLoss(temp_init=args.temp_init, bias_init=args.bias_init)

    # Move to device
    model.to(device)
    loss_fn.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=args.learning_rate
    )

    # Create datasets
    train_dataset_obj = ImageTextDataset(train_dataset, processor, args.final_resolution, args.dataset)
    val_dataset_obj = ImageTextDataset(val_dataset, processor, args.final_resolution, args.dataset)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset_obj,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset_obj,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Initialize history
    history = {
        'loss': [],
        'val_loss': [],
        'accuracy': [],
        'temp': [],
        'bias': [],
        'resolution': []
    }

    # Train for all epochs
    total_epochs = args.epochs_per_stage * len(args.resolutions)  # Match curriculum total epochs

    # Train model
    for epoch_group in range(len(args.resolutions)):
        logging.info(f"Starting epoch group {epoch_group+1}/{len(args.resolutions)}")
        history, best_val_loss = train_model(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            args=args,
            device=device,
            resolution=args.final_resolution,
            stage_name=f"Supervised Group {epoch_group+1}",
            history=history,
            checkpoint_dir=checkpoint_dir,
            processor=processor,
            dataset_type=args.dataset
        )

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "supervised_final_model.pth.tar")
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss_fn_state_dict': loss_fn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, final_model_path)
    logging.info(f"Final supervised model saved to {final_model_path}")

    return model, loss_fn, history

def run_curriculum_training(args, train_dataset, val_dataset, processor, device):
    """
    Run curriculum training with increasing resolutions
    """
    logging.info(f"Starting curriculum training with resolutions {args.resolutions}")

    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.checkpoint_dir, 'curriculum')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize model and loss function
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    loss_fn = SigLIPLoss(temp_init=args.temp_init, bias_init=args.bias_init)

    # Move to device
    model.to(device)
    loss_fn.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=args.learning_rate
    )

    # Initialize history
    history = {
        'loss': [],
        'val_loss': [],
        'accuracy': [],
        'temp': [],
        'bias': [],
        'resolution': []
    }

    # Train for each resolution
    for i, resolution in enumerate(args.resolutions):
        logging.info(f"Starting curriculum stage {i+1}/{len(args.resolutions)} with resolution {resolution}x{resolution}")

        # Create datasets for current resolution
        train_dataset_obj = ImageTextDataset(train_dataset, processor, resolution, args.dataset)
        val_dataset_obj = ImageTextDataset(val_dataset, processor, resolution, args.dataset)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset_obj,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )

        val_loader = DataLoader(
            val_dataset_obj,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )

        # Train model for this resolution
        history, best_val_loss = train_model(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            args=args,
            device=device,
            resolution=resolution,
            stage_name=f"Curriculum Stage {i+1}",
            history=history,
            checkpoint_dir=checkpoint_dir,
            processor=processor,
            dataset_type=args.dataset
        )

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "curriculum_final_model.pth.tar")
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss_fn_state_dict': loss_fn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, final_model_path)
    logging.info(f"Final curriculum model saved to {final_model_path}")

    return model, loss_fn, history

# Continue with the rest of your functions (plot_training_curves, calculate_metrics, etc.)
# The code above shows the key modifications needed to support smaller datasets

def main():
    """
    Main function to run the experiment
    """
    # Parse arguments
    args = parse_args()

    try:
        # Set up logging
        logger = setup_logging(args.log_dir)

        # Set random seed for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        logging.info(f"Using device: {device}")

        # Load processor
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Prepare datasets
        train_dataset, val_dataset = prepare_datasets(args, processor)

        # Run supervised training
        supervised_model, supervised_loss_fn, supervised_history = run_supervised_training(
            args, train_dataset, val_dataset, processor, device
        )

        # Run curriculum training
        curriculum_model, curriculum_loss_fn, curriculum_history = run_curriculum_training(
            args, train_dataset, val_dataset, processor, device
        )

        print("\n" + "="*50)
        print(f"EXPERIMENT COMPLETED SUCCESSFULLY USING {args.dataset} DATASET")
        print("="*50)
        print(f"Dataset: {args.dataset} with {args.dataset_size} samples")
        print(f"Final curriculum accuracy: {curriculum_history['accuracy'][-1]:.4f}")
        print(f"Final supervised accuracy: {supervised_history['accuracy'][-1]:.4f}")
        print("\nCheck the output directory for detailed results.")
        print("="*50)

    except Exception as e:
        logging.error(f"Error in experiment: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        print("\n" + "="*50)
        print("EXPERIMENT FAILED")
        print(f"Error: {str(e)}")
        print("Check logs for details.")
        print("="*50)

if __name__ == "__main__":
    main()
