#!/usr/bin/env python3
"""
Test script for GPU activation extraction
This script tests the activation extraction functionality with a small model
"""

import torch
import json
import os
from pathlib import Path
from rds6_config import get_experiment_path, ensure_rds6_dirs

def test_gpu_activation_extraction():
    """Test activation extraction with a small model on GPU."""
    print("🧪 Testing GPU activation extraction...")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires a GPU.")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Ensure RDS6 directories exist
    ensure_rds6_dirs()
    
    # Create a test experiment directory
    test_dir = "test_gpu_activations"
    rds6_checkpoints_dir = get_experiment_path(test_dir, "checkpoints")
    
    print(f"📁 RDS6 checkpoints directory: {rds6_checkpoints_dir}")
    
    # Create a small test model (much smaller than Llama for testing)
    print("🔧 Creating small test model...")
    
    # Simple transformer-like model for testing
    class TestModel(torch.nn.Module):
        def __init__(self, hidden_dim=64, num_layers=3):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            # Create layers similar to transformer structure
            self.layers = torch.nn.ModuleList()
            for i in range(num_layers):
                layer = torch.nn.ModuleDict({
                    'input_layernorm': torch.nn.LayerNorm(hidden_dim),
                    'self_attn': torch.nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True),
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
    
    # Create model and move to GPU
    model = TestModel(hidden_dim=64, num_layers=3)
    device = torch.device("cuda:0")
    model = model.to(device)
    
    print(f"✅ Model created and moved to GPU: {device}")
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create test input
    batch_size, seq_len = 2, 10
    test_input = torch.randn(batch_size, seq_len, 64).to(device)
    print(f"✅ Test input created: {test_input.shape} on {test_input.device}")
    
    # Run forward pass and capture activations
    print("🚀 Running forward pass...")
    with torch.no_grad():
        output, activations = model(test_input)
    
    print(f"✅ Forward pass completed. Output shape: {output.shape}")
    
    # Verify activations are on GPU
    print("\n🔍 Checking activation locations:")
    gpu_activations = 0
    total_activations = 0
    
    for category, layers in activations.items():
        for layer_name, activation in layers.items():
            total_activations += 1
            if activation.device.type == 'cuda':
                gpu_activations += 1
                print(f"  ✅ {category}/{layer_name}: GPU ({activation.shape})")
            else:
                print(f"  ❌ {category}/{layer_name}: {activation.device} ({activation.shape})")
    
    print(f"\n📊 GPU activations: {gpu_activations}/{total_activations}")
    
    # Test saving to RDS6 (move to CPU when saving)
    print("\n💾 Testing RDS6 storage...")
    
    test_id = "test_gpu_001"
    activation_file = f"{test_id}_activations.pt"
    activation_path = os.path.join(rds6_checkpoints_dir, activation_file)
    
    # Convert activations to numpy for storage (move to CPU)
    activations_to_save = {}
    for category, layers in activations.items():
        activations_to_save[category] = {}
        for layer_name, activation in layers.items():
            if activation is not None:
                # Move to CPU only when saving
                activation_cpu = activation.cpu().numpy()
                activations_to_save[category][layer_name] = activation_cpu
                print(f"  ✅ Moved {category}/{layer_name} to CPU: {activation_cpu.shape}")
    
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
            original = activations[category][layer_name].cpu().numpy()
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
            "hidden_dim": 64,
            "num_layers": 3,
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
    
    # Clean up
    del model, test_input, output, activations, activations_to_save
    torch.cuda.empty_cache()
    
    print(f"\n🎉 GPU activation extraction test completed!")
    print(f"📁 Test files created in: {rds6_checkpoints_dir}")
    
    if integrity_ok and gpu_activations == total_activations:
        print("✅ All tests passed! GPU activation extraction is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = test_gpu_activation_extraction()
    exit(0 if success else 1)
