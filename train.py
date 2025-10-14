import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import pandas as pd
import os
import matplotlib.pyplot as plt
from src.dataset import VariantTensorDataset
from src.model import GenomicClassifier, FocalLoss, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.font_manager as fm
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

fontPath = '/CTGlab/home/danilo/.fonts/HelveticaNeue-Medium.otf'
font_prop = fm.FontProperties(fname=fontPath, size=14)
fm.fontManager.addfont(fontPath)
# Set the default font size
mpl.rcParams['font.size'] = 16
mpl.rcParams['font.family'] = font_prop.get_name()
mpl.rcParams['font.sans-serif'] = font_prop.get_name()


def create_balanced_weights(labels: torch.Tensor) -> torch.Tensor:
    """Create class weights inversely proportional to class frequency"""
    class_counts = torch.bincount(labels.long())
    total_samples = len(labels)
    
    # Calculate weights: total_samples / (num_classes * class_count)
    weights = total_samples / (len(class_counts) * class_counts.float())
    return weights

def calculate_metrics(outputs, labels):
    """Calculate comprehensive metrics including F1 and AUC"""
    predicted = (torch.sigmoid(outputs) > 0.5).float()
    probabilities = torch.sigmoid(outputs)
    
    # Convert to numpy for sklearn metrics
    labels_np = labels.cpu().numpy().flatten()
    predicted_np = predicted.cpu().numpy().flatten()
    probabilities_np = probabilities.cpu().numpy().flatten()
    
    # Calculate metrics
    try:
        f1 = f1_score(labels_np, predicted_np, zero_division=0)
        auc = roc_auc_score(labels_np, probabilities_np) if len(np.unique(labels_np)) > 1 else 0
    except:
        f1, auc = 0, 0
        
    return f1, auc, predicted

if __name__ == '__main__':
    # Training parameters
    DATASET_DIR = "/CTGlab/projects/kreios/data/training/SRR15513228"
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    RANDOM_SEED = 42
    DROPOUT_RATE = 0.3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # Early stopping patience
    
    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load and Split Dataset ---
    full_dataset = VariantTensorDataset(dataset_dir=DATASET_DIR)
    labels = full_dataset.metadata['label'].values
    
    # Get input dimensions from first sample
    sample_input, _ = full_dataset[0]
    input_height, input_width = sample_input.shape[1], sample_input.shape[2]
    print(f"Input dimensions: {sample_input.shape} -> Height: {input_height}, Width: {input_width}")

    # Create indices and split them into training and validation sets
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    train_indices, val_indices = train_test_split(indices, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=labels)

    # Create Samplers for the DataLoaders
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create DataLoaders
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=4)

    print(f"Data loaded. Training on {len(train_indices)} samples, validating on {len(val_indices)} samples.")
    
    # Print class distribution
    train_labels = torch.tensor([labels[i] for i in train_indices])
    val_labels = torch.tensor([labels[i] for i in val_indices])
    print(f"Training class distribution: {torch.bincount(train_labels.long())}")
    print(f"Validation class distribution: {torch.bincount(val_labels.long())}")

    # --- Model Setup with Improved Architecture ---
    model = GenomicClassifier(
        input_height=input_height,
        input_width=input_width,
        compressed_feature_size=128,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    # Calculate class weights for handling imbalance
    class_weights = create_balanced_weights(train_labels)
    pos_weight = class_weights[1] / class_weights[0]
    print(f"Class weights: {class_weights}, pos_weight: {pos_weight:.3f}")
    
    # Choose loss function - Focal Loss for severe imbalance
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    # Alternative: Weighted BCE Loss
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.001)

    # --- Initialize tracking lists for plotting ---
    train_losses = []
    val_losses = []
    train_recall_0, train_recall_1 = [], []
    val_recall_0, val_recall_1 = [], []
    train_f1_scores, val_f1_scores = [], []
    train_auc_scores, val_auc_scores = [], []
    learning_rates = []

    # --- Training and Validation Loop ---
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} (LR: {current_lr:.2e}) ---")
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        train_correct_0, train_total_0 = 0, 0
        train_correct_1, train_total_1 = 0, 0
        all_train_outputs, all_train_labels = [], []

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Store outputs and labels for comprehensive metrics
            all_train_outputs.append(outputs.detach())
            all_train_labels.append(labels.detach())
            
            # Track stratified predictions
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            mask_0 = (labels == 0).squeeze()
            if mask_0.any():
                train_total_0 += mask_0.sum().item()
                train_correct_0 += ((predicted == labels) & (labels == 0)).sum().item()
            
            mask_1 = (labels == 1).squeeze()
            if mask_1.any():
                train_total_1 += mask_1.sum().item()
                train_correct_1 += ((predicted == labels) & (labels == 1)).sum().item()

        # Calculate training metrics
        epoch_loss = running_loss / len(train_indices)
        recall_0 = train_correct_0 / train_total_0 if train_total_0 > 0 else 0
        recall_1 = train_correct_1 / train_total_1 if train_total_1 > 0 else 0
        
        # Calculate comprehensive training metrics
        all_train_outputs = torch.cat(all_train_outputs)
        all_train_labels = torch.cat(all_train_labels)
        train_f1, train_auc, _ = calculate_metrics(all_train_outputs, all_train_labels)
        
        # Store training metrics
        train_losses.append(epoch_loss)
        train_recall_0.append(recall_0 * 100)
        train_recall_1.append(recall_1 * 100)
        train_f1_scores.append(train_f1 * 100)
        train_auc_scores.append(train_auc * 100)
        
        print(f"Train Loss: {epoch_loss:.4f} | F1: {train_f1:.3f} | AUC: {train_auc:.3f}")
        print(f"Training: {train_correct_1}/{train_total_1} for 1, {train_correct_0}/{train_total_0} for 0")
        print(f"Training Recall: Class 0: {recall_0:.3f}, Class 1: {recall_1:.3f}")

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct_0, val_total_0 = 0, 0
        val_correct_1, val_total_1 = 0, 0
        all_val_outputs, all_val_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1).float()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                # Store outputs and labels for comprehensive metrics
                all_val_outputs.append(outputs)
                all_val_labels.append(labels)
                
                # Track stratified predictions
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                
                mask_0 = (labels == 0).squeeze()
                if mask_0.any():
                    val_total_0 += mask_0.sum().item()
                    val_correct_0 += ((predicted == labels) & (labels == 0)).sum().item()
                
                mask_1 = (labels == 1).squeeze()
                if mask_1.any():
                    val_total_1 += mask_1.sum().item()
                    val_correct_1 += ((predicted == labels) & (labels == 1)).sum().item()

        # Calculate validation metrics
        epoch_val_loss = val_loss / len(val_indices)
        val_recall_0_epoch = val_correct_0 / val_total_0 if val_total_0 > 0 else 0
        val_recall_1_epoch = val_correct_1 / val_total_1 if val_total_1 > 0 else 0
        
        # Calculate comprehensive validation metrics
        all_val_outputs = torch.cat(all_val_outputs)
        all_val_labels = torch.cat(all_val_labels)
        val_f1, val_auc, _ = calculate_metrics(all_val_outputs, all_val_labels)
        
        # Store validation metrics
        val_losses.append(epoch_val_loss)
        val_recall_0.append(val_recall_0_epoch * 100)
        val_recall_1.append(val_recall_1_epoch * 100)
        val_f1_scores.append(val_f1 * 100)
        val_auc_scores.append(val_auc * 100)
        
        print(f"Validation Loss: {epoch_val_loss:.4f} | F1: {val_f1:.3f} | AUC: {val_auc:.3f}")
        print(f"Validation: {val_correct_1}/{val_total_1} for 1, {val_correct_0}/{val_total_0} for 0")
        print(f"Validation Recall: Class 0: {val_recall_0_epoch:.3f}, Class 1: {val_recall_1_epoch:.3f}")
        
        # Learning rate scheduling
        scheduler.step(epoch_val_loss)
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved! Val Loss: {best_val_loss:.4f}")
        
        # Early stopping check
        if early_stopping(epoch_val_loss):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print("\n--- Training Complete ---")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    print("Best model loaded for final evaluation")

    # --- Enhanced Plotting ---
    epochs_range = range(1, len(train_losses) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Loss over epochs
    ax1.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Recall percentage over epochs for each class
    ax2.plot(epochs_range, train_recall_0, 'b--', label='Train Recall Class 0', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs_range, train_recall_1, 'b-', label='Train Recall Class 1', linewidth=2, marker='s', markersize=4)
    ax2.plot(epochs_range, val_recall_0, 'r--', label='Val Recall Class 0', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs_range, val_recall_1, 'r-', label='Val Recall Class 1', linewidth=2, marker='s', markersize=4)
    ax2.set_title('Recall Percentage by Class', fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Recall (%)')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: F1 Score over epochs
    ax3.plot(epochs_range, train_f1_scores, 'b-', label='Training F1', linewidth=2)
    ax3.plot(epochs_range, val_f1_scores, 'r-', label='Validation F1', linewidth=2)
    ax3.set_title('F1 Score Over Epochs', fontweight='bold')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('F1 Score (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: AUC Score over epochs
    ax4.plot(epochs_range, train_auc_scores, 'b-', label='Training AUC', linewidth=2)
    ax4.plot(epochs_range, val_auc_scores, 'r-', label='Validation AUC', linewidth=2)
    ax4.set_title('AUC Score Over Epochs', fontweight='bold')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('AUC Score (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_training_metrics.png', dpi=300, bbox_inches='tight')
    print("Enhanced plots saved as 'enhanced_training_metrics.png'")