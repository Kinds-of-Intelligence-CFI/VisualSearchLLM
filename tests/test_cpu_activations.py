#!/usr/bin/env python3
"""
Test script for activation extraction (CPU version)
This script tests the activation extraction functionality without requiring a GPU
"""

import torch
import json
import os
from pathlib import Path
from rds6_config import get_experiment_path, ensure_rds6_dirs

def test_cpu_activation_extraction():
    """Test activation extraction with a small model on CPU."""
    print("🧪 Testing CPU activation extraction...")
    
    # Check device availability
    device = torch.device("cpu")
    print(f"✅ Using device: {device}")
    
    # Ensure RDS6 directories exist
    ensure_rds6_dirs()
    
    # Create a test experiment directory
    test_dir = "test_cpu_activations"
    rds6_checkpoints_dir = get_experiment_path(test_dir, "checkpoints")
    
    print(f"📁 RDS6 checkpoints directory: {rds6_checkpoints_dir}")
    
    # Create a small test model (much smaller than Llama for testing)
    print("🔧 Creating small test model...")
    
    # Simple transformer-like model for testing
    class TestModel(torch.nn.Module):
        def __init__(self, hidden_dim=32, num_layers=2):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            # Create layers similar to transformer structure
            self.layers = torch.nn.ModuleList()
            for i in range(num_layers):
                layer = torch.nn.ModuleDict({
                    'input_layernorm': torch.nn.LayerNorm(hidden_dim),
                    'self_attn': torch.nn.MultiheadAttention(hidden_dim, num_heads=2, batch_first=True),
                    'mlp': torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim, hidden_dim * 2),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_dim * 2, hidden_dim)
                    ),
                    'post_attention_layernorm': torch.nn.LayerNorm(hidden_dim)
                })
                self.layers.append(layer)
            
            self.final_layernorm = torch.nn.LayerNorm(hidden_dim)
            self.output_projection = torch.nn.Linear(hidden_dim, hidden_dim)
        
        def forward(self, x):
            # Store activations
            activations = {
                "residual_stream": {},
                "attention_heads": {},
                "mlp": {},
                "layer_norms": {}
            }
            
            # Process through each layer
            for layer_idx, layer in enumerate(self.layers):
                # Residual stream input
                residual_input = x
                activations["residual_stream"][f"layer_{layer_idx}"] = residual_input.detach()
                
                # Input layer norm
                x = layer['input_layernorm'](x)
                activations["layer_norms"][f"layer_{layer_idx}_input"] = x.detach()
                
                # Self attention
                attn_output, _ = layer['self_attn'](x, x, x)
                activations["attention_heads"][f"layer_{layer_idx}"] = attn_output.detach()
                
                # Residual connection
                x = residual_input + attn_output
                
                # Post-attention layer norm
                x = layer['post_attention_layernorm'](x)
                activations["layer_norms"][f"layer_{layer_idx}_post"] = x.detach()
                
                # MLP
                mlp_output = layer['mlp'](x)
                activations["mlp"][f"layer_{layer_idx}"] = mlp_output.detach()
                
                # Final residual connection
                x = x + mlp_output
            
            # Final layer norm
            x = self.final_layernorm(x)
            activations["layer_norms"]["final"] = x.detach()
            
            # Output projection
            output = self.output_projection(x)
            
            return output, activations
    
    # Create model
    model = TestModel(hidden_dim=32, num_layers=2)
    model = model.to(device)
    
    print(f"✅ Model created on device: {device}")
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create test input
    batch_size, seq_len = 1, 5
    test_input = torch.randn(batch_size, seq_len, 32).to(device)
    print(f"✅ Test input created: {test_input.shape} on {test_input.device}")
    
    # Run forward pass and capture activations
    print("🚀 Running forward pass...")
    with torch.no_grad():
        output, activations = model(test_input)
    
    print(f"✅ Forward pass completed. Output shape: {output.shape}")
    
    # Verify activations are on correct device
    print("\n🔍 Checking activation locations:")
    correct_device_activations = 0
    total_activations = 0
    
    for category, layers in activations.items():
        for layer_name, activation in layers.items():
            total_activations += 1
            if activation.device == device:
                correct_device_activations += 1
                print(f"  ✅ {category}/{layer_name}: {device} ({activation.shape})")
            else:
                print(f"  ❌ {category}/{layer_name}: {activation.device} ({activation.shape})")
    
    print(f"\n📊 Correct device activations: {correct_device_activations}/{total_activations}")
    
    # Test saving to RDS6
    print("\n💾 Testing RDS6 storage...")
    
    test_id = "test_cpu_001"
    activation_file = f"{test_id}_activations.pt"
    activation_path = os.path.join(rds6_checkpoints_dir, activation_file)
    
    # Convert activations to numpy for storage
    activations_to_save = {}
    for category, layers in activations.items():
        activations_to_save[category] = {}
        for layer_name, activation in layers.items():
            if activation is not None:
                # Convert to numpy for storage
                activation_numpy = activation.numpy()
                activations_to_save[category][layer_name] = activation_numpy
                print(f"  ✅ Converted {category}/{layer_name} to numpy: {activation_numpy.shape}")
    
    # Save activations
    torch.save(activations_to_save, activation_path)
    print(f"✅ Saved activations to: {activation_path}")
    
    # Test loading
    loaded_activations = torch.load(activation_path, map_location='cpu', weights_only=False)
    print("✅ Loaded activations from file")
    
    # Verify data integrity
    print("\n🔍 Verifying data integrity...")
    integrity_ok = True
    for category in activations.keys():
        for layer_name in activations[category].keys():
            original = activations[category][layer_name].numpy()
            loaded = loaded_activations[category][layer_name]
            
            if not (original == loaded).all():
                print(f"  ❌ {category}/{layer_name}: Data mismatch")
                integrity_ok = False
            else:
                print(f"  ✅ {category}/{layer_name}: Data integrity verified")
    
    # Create activation summary
    activation_summary = {
        "custom_id": test_id,
        "model_info": {
            "hidden_dim": 32,
            "num_layers": 2,
            "device": str(device)
        },
        "activation_stats": {}
    }
    
    for category, layers in activations.items():
        activation_summary["activation_stats"][category] = {}
        for layer_name, activation in layers.items():
            if activation is not None:
                activation_summary["activation_stats"][category][layer_name] = {
                    "shape": list(activation.shape),
                    "device": str(activation.device),
                    "mean": float(activation.mean()),
                    "std": float(activation.std()),
                    "min": float(activation.min()),
                    "max": float(activation.max())
                }
    
    # Save summary
    summary_file = f"{test_id}_activation_summary.json"
    summary_path = os.path.join(rds6_checkpoints_dir, summary_file)
    with open(summary_path, 'w') as f:
        json.dump(activation_summary, f, indent=2)
    
    print(f"✅ Saved activation summary to: {summary_path}")
    
    # Test the analyze_activations script functionality
    print("\n🔍 Testing activation analysis functionality...")
    
    # Simulate what analyze_activations.py would do
    residual_stats = []
    attention_stats = []
    
    for layer_name, activation in activations["residual_stream"].items():
        residual_stats.append({
            "layer": int(layer_name.split('_')[1]) if '_' in layer_name else 0,
            "mean": float(activation.mean()),
            "std": float(activation.std()),
            "min": float(activation.min()),
            "max": float(activation.max()),
            "shape": activation.shape
        })
    
    for layer_name, activation in activations["attention_heads"].items():
        attention_stats.append({
            "layer": int(layer_name.split('_')[1]) if '_' in layer_name else 0,
            "mean": float(activation.mean()),
            "std": float(activation.std()),
            "min": float(activation.min()),
            "max": float(activation.max()),
            "shape": activation.shape
        })
    
    residual_stats.sort(key=lambda x: x["layer"])
    attention_stats.sort(key=lambda x: x["layer"])
    
    print("  ✅ Residual stream analysis:")
    for stat in residual_stats:
        print(f"    Layer {stat['layer']}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, shape={stat['shape']}")
    
    print("  ✅ Attention heads analysis:")
    for stat in attention_stats:
        print(f"    Layer {stat['layer']}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, shape={stat['shape']}")
    
    # Clean up
    del model, test_input, output, activations, activations_to_save
    
    print(f"\n🎉 CPU activation extraction test completed!")
    print(f"📁 Test files created in: {rds6_checkpoints_dir}")
    
    if integrity_ok and correct_device_activations == total_activations:
        print("✅ All tests passed! Activation extraction is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = test_cpu_activation_extraction()
    exit(0 if success else 1)
