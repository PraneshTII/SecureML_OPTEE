#!/usr/bin/env python3
"""
Quick script to generate .importance.txt for downloaded victim models
Uses the existing TensorShield codebase structure
"""

import torch
import os
import os.path as osp
import sys
import copy
import torch.optim as optim

# Add paths
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

# Import from existing codebase
from knockoff import datasets
import knockoff.models.cifar as cifar_models
import knockoff.models.imagenet as imagenet_models


# ========== CPU-Compatible get_importance functions ==========

def load_target_model_cpu(target_model, victim_dir, device='cpu'):
    """Load target model weights - CPU compatible version"""
    target_path = osp.join(victim_dir, "checkpoint.pth.tar")
    target_model = target_model.to(device)
    print(f"Load model from {target_path}")
    checkpoint = torch.load(target_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    target_model.load_state_dict(state_dict, strict=False)
    target_model.eval()
    return target_model


def get_protect_layers_cpu(train_dataset, target_model, percent, is_random, victim_dir, device='cpu'):
    """Get the most important layers to protect - CPU compatible version"""
    
    info_path = osp.join(victim_dir, '.importance.txt')
    
    if osp.exists(info_path):
        print('Importance info exists. Loading...')
        with open(info_path, 'r') as f:
            protect_list = f.read().splitlines()[0].split(',')
        print('Importance info loaded.')
        
        # Calculate how many to protect
        import random
        n = len(protect_list)
        num_elements = int(n * percent)
        print(f"get_sample_layers n={n}  num_select={num_elements}")
        
        if is_random:
            print("get_sample_layers is_random")
            return random.sample(protect_list, num_elements)
        else:
            print("get_sample_layers not_random")
            return protect_list[:num_elements]
    
    print('Importance info does not exist. Calculating...')
    
    # Adjust num_workers for CPU
    num_workers = 0
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers)
    
    # Split validation
    total_train_examples = len(train_dataset)
    val_size = int(total_train_examples * 0.1)
    train_size = total_train_examples - val_size
    train_subset, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=num_workers)
    
    pretrained = copy.deepcopy(target_model)
    model = copy.deepcopy(pretrained)
    model = load_target_model_cpu(model, victim_dir, device=device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_function = torch.nn.CrossEntropyLoss()
    
    weight_updates = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
    weight_importance = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
    
    print(f"Training for importance calculation on {device}...")
    for epoch in range(10):
        print(f"  Epoch {epoch+1}/10")
        batch_count = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        current_update = (-optimizer.param_groups[0]['lr'] * param.grad).abs()
                        param_grad_abs = param.grad.abs()
                        weight_updates[name].add_(current_update)
                        weight_importance[name].addcmul_(current_update, param_grad_abs)
                        weight_importance[name].div_(param.numel())
            
            batch_count += 1
            if batch_count % 50 == 0:
                print(f"    Processed {batch_count} batches, Loss: {loss.item():.4f}")
    
    sorted_importances = {k: v.sum().item() for k, v in weight_importance.items()}
    sorted_importances = sorted(sorted_importances.items(), key=lambda item: item[1], reverse=True)
    print('--------------------------------------------')
    print(f'sorted_importances: {sorted_importances}')
    print('--------------------------------------------')
    
    filtered_importances = {k: v.sum().item() for k, v in weight_importance.items() if 'bn' not in k and 'downsample' not in k}
    sorted_filtered_importances = sorted(filtered_importances.items(), key=lambda item: item[1], reverse=True)
    print('--------------------------------------------')
    print(f'sorted_filtered_importances: {sorted_filtered_importances}')
    print('--------------------------------------------')
    
    new_select = [k for k in sorted_importances if "weight" in k[0]]
    with open(info_path, 'w') as f:
        f.write(','.join([item[0] for item in new_select]))
    
    import random
    n = len(new_select)
    num_elements = int(n * percent)
    print(f"get_sample_layers n={n}  num_select={num_elements}")
    
    if is_random:
        selected = random.sample(new_select, num_elements)
    else:
        selected = new_select[:num_elements]
    
    protect_list = [item[0] for item in selected]
    
    print('--------------------------------------------')
    print(f'protect_list: {protect_list}')
    print('--------------------------------------------')
    return protect_list


# ========== Main Generation Function ==========

def generate_importance(victim_dir, dataset_name, arch, num_classes):
    """
    Generate .importance.txt using CPU-compatible get_importance
    """
    print(f"\n{'='*70}")
    print(f"Processing: {victim_dir}")
    print(f"{'='*70}\n")
    
    # Check if already exists
    importance_path = osp.join(victim_dir, '.importance.txt')
    if osp.exists(importance_path):
        print(f"✓ .importance.txt already exists!")
        with open(importance_path, 'r') as f:
            layers = f.read().strip().split(',')
        print(f"  Layers: {len(layers)}")
        print(f"  Top 3: {layers[:3]}")
        return True
    
    # Setup
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = datasets.__dict__[dataset_name]
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    trainset = dataset(train=True, transform=train_transform)
    
    print(f"Dataset: {dataset_name} ({len(trainset)} samples)")
    
    # Load model architecture based on what exists in the codebase
    print(f"Loading architecture: {arch}")
    
    if arch == 'alexnet':
        from knockoff.models.cifar import alexnet
        model = alexnet(num_classes=num_classes)
    elif arch == 'vgg16_bn':
        from knockoff.models.cifar import vgg16_bn
        model = vgg16_bn(num_classes=num_classes)
    elif arch == 'resnet18':
        from knockoff.models.imagenet import resnet18
        model = resnet18(pretrained='imagenet')  # Use 'imagenet' string
        model.last_linear = torch.nn.Linear(model.last_linear.in_features, num_classes)
    elif arch == 'resnet50':
        from knockoff.models.imagenet import resnet50
        model = resnet50(pretrained='imagenet')  # Use 'imagenet' string
        model.last_linear = torch.nn.Linear(model.last_linear.in_features, num_classes)
    elif arch == 'mobilenetv2':
        from knockoff.models.imagenet import mobilenetv2
        model = mobilenetv2(pretrained='imagenet')  # Use 'imagenet' string, not False
        model.last_linear = torch.nn.Linear(model.last_linear.in_features, num_classes)
    else:
        print(f"✗ Unsupported architecture: {arch}")
        return False
    
    # Load checkpoint
    checkpoint_path = osp.join(victim_dir, 'checkpoint.pth.tar')
    if not osp.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Warning: {e}")
        print("Continuing anyway...")
    
    model = model.to(device)
    
    # Count layers
    layer_names = [n for n, p in model.named_parameters() if 'weight' in n]
    print(f"Total weight layers: {len(layer_names)}")
    
    # Generate importance using CPU function
    print("\nGenerating importance scores...")
    print("This will take 10-15 minutes (training for 10 epochs)...")
    print("Running on CPU - please be patient!")
    
    try:
        protect_layers = get_protect_layers_cpu(
            trainset,
            model,
            0.5,  # Dummy value, just need to generate the file
            False,
            victim_dir,
            device=device
        )
        
        print(f"\n✓ SUCCESS! Importance generated")
        print(f"✓ Saved to: {importance_path}")
        
        # Display results
        with open(importance_path, 'r') as f:
            layers = f.read().strip().split(',')
        
        print(f"\nTop 5 most important layers:")
        for i, layer in enumerate(layers[:5], 1):
            print(f"  {i}. {layer}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main function - define your models here
    """
    
    # Define models you downloaded
    models = [
        ('models/victim/cifar10-alexnet', 'CIFAR10', 'alexnet', 10),
        ('models/victim/cifar10-mobilenetv2', 'CIFAR10', 'mobilenetv2', 10),
        # Add more as you download them:
        # ('models/victim/cifar10-resnet18', 'CIFAR10', 'resnet18', 10),
        # ('models/victim/cifar100-resnet50', 'CIFAR100', 'resnet50', 100),
    ]
    
    print("="*70)
    print("TensorShield - Generate Importance for Victim Models")
    print("="*70)
    
    results = []
    for victim_dir, dataset, arch, num_classes in models:
        if not osp.exists(victim_dir):
            print(f"\n✗ Skipping {victim_dir} (not found)")
            continue
        
        success = generate_importance(victim_dir, dataset, arch, num_classes)
        results.append((victim_dir, success))
    
    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for victim_dir, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {victim_dir}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
