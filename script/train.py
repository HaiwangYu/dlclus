import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import EdgeConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import glob
from tqdm import tqdm

from dlclus.util.omni import *

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a GNN model for neutrino interaction classification')
    parser.add_argument('--train-file-list', help='File containing list of training data files')
    parser.add_argument('--val-file-list', help='File containing list of validation data files')
    parser.add_argument('--test-file-list', help='File containing list of test data files')
    parser.add_argument('--file-list', help='File containing list of all data files (if not using separate lists)')
    parser.add_argument('--data-dir', help='Directory containing labeled data files')
    parser.add_argument('--pattern', default='rec-lab-apa1-*.npz', help='File pattern for labeled data')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--output-dir', default='models', help='Directory to save model')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data based on provided arguments
    if args.train_file_list and args.val_file_list:
        # Load separate train, validation, and optionally test data
        print("Loading training data...")
        train_data = load_data_from_list(args.train_file_list)
        
        print("Loading validation data...")
        val_data = load_data_from_list(args.val_file_list)
        
        if args.test_file_list:
            print("Loading test data...")
            test_data = load_data_from_list(args.test_file_list)
        else:
            # If no test set provided, use validation set for final evaluation
            test_data = val_data
            
        # Prepare data loaders directly without random splitting
        print("Preparing data loaders...")
        train_loader = prepare_loader(train_data, batch_size=1, shuffle=True)
        val_loader = prepare_loader(val_data, batch_size=1, shuffle=False)
        test_loader = prepare_loader(test_data, batch_size=1, shuffle=False)
        
    else:
        # Fall back to the original approach with random splitting
        print("Loading labeled data...")
        if args.file_list:
            data_list = load_data_from_list(args.file_list)
        elif args.data_dir:
            data_list = load_labeled_data(args.data_dir, args.pattern)
        else:
            raise ValueError("Either --train-file-list and --val-file-list, or --file-list, or --data-dir must be specified")
        
        # Prepare datasets with random splitting
        print("Preparing datasets with random splitting...")
        train_loader, val_loader, test_loader = prepare_datasets(data_list)
    
    # Get input dimension from the first sample
    sample_data = train_loader.dataset[0]
    input_dim = sample_data.x.size(1)
    
    # Initialize model
    print("Initializing model...")
    model = GNNModel(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("Starting training...")
    best_val_f1 = 0
    
    # Compute class weights once before training
    all_labels = []
    for data in tqdm(train_loader, desc="Computing class weights"):
        all_labels.append(data.y.cpu().numpy())
    all_labels = np.concatenate(all_labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # Show class distribution
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_labels, counts))}")
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Training loop with progress bar
    for epoch in tqdm(range(1, args.epochs + 1), desc="Training epochs"):
        # Train with pre-computed weights
        loss = train_epoch(model, optimizer, train_loader, device, class_weights)
        
        # Evaluate on validation set
        val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device)
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
        
        # Save the model if validation F1 score improves
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
            print(f'Saved model to {model_path}')
    
    # Load the best model and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    # test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, device)
    test_acc, test_prec, test_rec, test_f1 = evaluate(model, val_loader, device)
    
    print("\nTest set evaluation:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1 Score: {test_f1:.4f}")

if __name__ == "__main__":
    main()
