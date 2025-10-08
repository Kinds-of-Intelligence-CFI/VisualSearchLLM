import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from rds6_config import get_experiment_path, get_checkpoint_path

def load_model_and_run(directory, model_choice, batch_file, output_file):
    # GPU optimization setup
    if torch.cuda.is_available():
        # Test CUDA functionality first
        print("🧪 Testing CUDA functionality...")
        test_tensor = torch.randn(100, 100).cuda()
        test_result = torch.mm(test_tensor, test_tensor)
        print(f"✅ CUDA test passed: {test_result.shape}")
        del test_tensor, test_result
        torch.cuda.empty_cache()
        

        # Set memory fraction and optimization
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        
        print(f"GPU Optimization enabled:")
        print(f"  - TF32: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  - cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  - Available GPUs: {torch.cuda.device_count()}")
        print(f"  - Current GPU: {torch.cuda.current_device()}")
        print(f"  - GPU Name: {torch.cuda.get_device_name()}")
        print(f"  - GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("No CUDA available - will run on CPU (very slow!)")
    
    # Select the appropriate model based on the choice
    if model_choice == "llama11B":
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    elif model_choice == "llama90B":
        model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"
    else:
        raise ValueError(f"Unsupported model choice: {model_choice}")
    
    print(f"Loading model: {model_id}", flush=True)
    actualDirectory="results/"+directory
    
    # Create RDS6 paths for this experiment
    rds6_results_dir = get_experiment_path(directory, "results")
    rds6_checkpoints_dir = get_experiment_path(directory, "checkpoints")
    
    print(f"RDS6 Results dir: {rds6_results_dir}")
    print(f"RDS6 Checkpoints dir: {rds6_checkpoints_dir}")
    
    # Load model and processor with GPU optimizations
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    
    # Load model directly to GPU (no device_map="auto" to avoid conflicts)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=os.getenv("HF_TOKEN"),
    )
    model.tie_weights()
    print("Loaded Model", flush=True)
    
    processor = AutoProcessor.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))
    print("Loaded Processor", flush=True)
    
    # Debug: sharding and attention backend
    is_sharded = hasattr(model, "hf_device_map")
    print("is_sharded:", is_sharded)
    if is_sharded:
        try:
            print("devices:", sorted(set(model.hf_device_map.values())))
        except Exception:
            pass
    print("attn_impl:", getattr(model.config, "_attn_implementation", None))
    
    # Construct full file paths
    input_path = os.path.join(actualDirectory, batch_file)
    output_path = os.path.join(actualDirectory, output_file)
    
    print(f"Processing batch file: {input_path}", flush=True)
    print(f"Writing results to: {output_path}", flush=True)
    
    # Process each line in the input file
    with open(input_path, 'r') as f, open(output_path, 'w') as out_f:
        for i, line in enumerate(f):
            print("begin image "+str(i), flush=True)
            line = line.strip()
            if not line:
                continue  # Skip empty lines
                
            record = json.loads(line)
            custom_id = record.get("custom_id")
            
            # Get messages
            messages = record.get("messages")
            
            # Load the image
            image_path = os.path.join(actualDirectory, custom_id)
            image = Image.open(image_path)


            if image is None:
                raise ValueError(f"Failed to load image at path: {image_path}")
            if not hasattr(image, "size") or not hasattr(image, "mode"):
                raise ValueError(f"Loaded object is not a valid image: {image_path}")
            print(f"image info: {image.mode} {image.size}", flush=True)
            
            # Process the input (ensure correct multimodal formatting and device placement)
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

            print(f"Input Text: {input_text}, {type(input_text)}")


            inputs = processor(
                images=image,
                text=input_text,
                add_special_tokens=False, 
                return_tensors="pt"
            )

            
            prompt_length = inputs["input_ids"].shape[-1]
            last_token_index = max(0, prompt_length - 1)
            # Generate the output
            output = model.generate(**inputs, max_new_tokens=30)
            
            # Remove the prompt tokens from the output
            new_tokens = output[0][prompt_length:]
            result = processor.decode(new_tokens, skip_special_tokens=True)

            # Initialize storage for all activations
            all_activations = {
                "residual_stream": {},
                "attention_heads": {},
                "mlp": {},
                "layer_norms": {},
                "vision_local_residual": {},
                "vision_global_residual": {},
                "projector_input": {},
                "projector_output": {},
            }

            hooks = []
            # Track whether we've already captured the prefill (prompt) state per layer/category
            captured_flags = {
                "residual_stream": set(),
                "attention_heads": set(),
                "mlp": set(),
                "layer_norms": set(),
                "vision_local_residual": set(),
                "vision_global_residual": set(),
                "projector_input": set(),
                "projector_output": set(),
            }

            
            # Pre-forward hook: captures the input (residual stream) before the layer runs
            def create_pre_activation_hook(layer_idx):
                def hook(module, input):
                    key = f"layer_{layer_idx}"
                    if key in captured_flags["residual_stream"]:
                        return
                    try:
                        # input[0]: [batch, seq_len, hidden] at layer entry
                        x = input[0]
                        if torch.is_tensor(x) and x.dim() >= 2:
                            sl = x[:, last_token_index:last_token_index+1, :]
                            all_activations["residual_stream"][key] = sl.detach()
                            captured_flags["residual_stream"].add(key)
                    except Exception:
                        pass
                return hook

            # Forward hook: captures the output of attention/MLP/layer norms
            def create_forward_activation_hook(layer_idx, activation_type):
                def hook(module, inputs, output):
                    key = f"layer_{layer_idx}"
                    if key in captured_flags[activation_type]:
                        return
                    try:
                        y = output[0] if isinstance(output, (tuple, list)) else output
                        if torch.is_tensor(y) and y.dim() >= 2:
                            sl = y[:, last_token_index:last_token_index+1, :]
                            all_activations[activation_type][key] = sl.detach()
                            captured_flags[activation_type].add(key)
                    except Exception:
                        pass
                return hook
            
            # Version-agnostic access to language decoder layers
            try:
                decoder = model.language_model.get_decoder()
                language_layers = decoder.layers
                layer_path = "model.language_model.get_decoder().layers"
            except Exception:
                try:
                    language_layers = model.language_model.model.layers
                    layer_path = "model.language_model.model.layers"
                except Exception:
                    language_layers = model.model.language_model.layers
                    layer_path = "model.model.language_model.layers"
          
            total_layers = len(language_layers)
            print(f"Found layers at: {layer_path}")
            print(f"Total layers: {total_layers}")
            
            # Select 3 strategic layers: early, middle, and late (clamped & deduped)
            candidates = [5, total_layers // 2, total_layers - 5]
            selected_layers = sorted({max(0, min(total_layers - 1, idx)) for idx in candidates})
            print(f"Selecting 3 strategic layers out of {total_layers} total layers: {selected_layers}")
            
            for layer_idx in selected_layers:
                layer = language_layers[layer_idx]
                # Hook residual stream (input to the layer)
                hooks.append(layer.register_forward_pre_hook(
                    create_pre_activation_hook(layer_idx)
                ))
                
                # Hook attention output - check if it's self-attention or cross-attention
                if hasattr(layer, 'self_attn'):
                    # Self-attention layer
                    hooks.append(layer.self_attn.register_forward_hook(
                        create_forward_activation_hook(layer_idx, "attention_heads")
                    ))
                elif hasattr(layer, 'cross_attn'):
                    # Cross-attention layer
                    hooks.append(layer.cross_attn.register_forward_hook(
                        create_forward_activation_hook(layer_idx, "attention_heads")
                    ))
                
                # Hook MLP output
                hooks.append(layer.mlp.register_forward_hook(
                    create_forward_activation_hook(layer_idx, "mlp")
                ))
                
                # Hook layer norm outputs - check which type of layer norm
                if hasattr(layer, 'input_layernorm'):
                    hooks.append(layer.input_layernorm.register_forward_hook(
                        create_forward_activation_hook(layer_idx, "layer_norms")
                    ))
                if hasattr(layer, 'post_attention_layernorm'):
                    hooks.append(layer.post_attention_layernorm.register_forward_hook(
                        create_forward_activation_hook(layer_idx, "layer_norms")
                    ))

            # ---- Vision hooks: residual streams (minimal mech-interp capture) ----
            def create_vision_pre_activation_hook(layer_idx, category_name):
                def hook(module, input):
                    key = f"layer_{layer_idx}"
                    if key in captured_flags[category_name]:
                        return
                    try:
                        x = input[0]
                        # For vision encoders, take the first class token of the first tile: index 0
                        if torch.is_tensor(x) and x.dim() >= 3:
                            sl = x[:, 0:1, :]
                            all_activations[category_name][key] = sl.detach()
                            captured_flags[category_name].add(key)
                    except Exception:
                        pass
                return hook

            # Select early/mid/late for local encoder
            try:
                vision_local_layers = model.vision_model.transformer.layers
                n_local = len(vision_local_layers)
                local_candidates = [max(0, min(n_local - 1, 5)), n_local // 2, max(0, n_local - 5)]
                local_selected = sorted({idx for idx in local_candidates})
                print(f"Vision local encoder layers: {n_local}, capturing at {local_selected}")
                for v_idx in local_selected:
                    v_layer = vision_local_layers[v_idx]
                    hooks.append(v_layer.register_forward_pre_hook(
                        create_vision_pre_activation_hook(v_idx, "vision_local_residual")
                    ))
            except Exception as e:
                print(f"Vision local hook setup skipped: {e}")

            # Select early/mid/late for global encoder
            try:
                vision_global_layers = model.vision_model.global_transformer.layers
                n_global = len(vision_global_layers)
                global_candidates = [max(0, min(n_global - 1, 2)), n_global // 2, max(0, n_global - 2)]
                global_selected = sorted({idx for idx in global_candidates})
                print(f"Vision global encoder layers: {n_global}, capturing at {global_selected}")
                for v_idx in global_selected:
                    v_layer = vision_global_layers[v_idx]
                    hooks.append(v_layer.register_forward_pre_hook(
                        create_vision_pre_activation_hook(v_idx, "vision_global_residual")
                    ))
            except Exception as e:
                print(f"Vision global hook setup skipped: {e}")

            # ---- Projector hooks: capture projector input and output (sliced) ----
            def projector_forward_hook(module, inputs, output):
                key = "projector"
                try:
                    if "projector_input" not in captured_flags:
                        pass
                    # Input: vision hidden states before projection (likely 5D)
                    vin = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
                    if torch.is_tensor(vin):
                        if vin.dim() == 5:
                            vin_slice = vin[:, 0:1, 0:1, 0:1, :]
                        elif vin.dim() >= 3:
                            vin_slice = vin[:, 0:1, :]
                        else:
                            vin_slice = vin
                        if "input" not in all_activations["projector_input"]:
                            all_activations["projector_input"]["input"] = vin_slice.detach()
                            captured_flags["projector_input"].add(key)
                    # Output: projected states
                    vout = output[0] if isinstance(output, (tuple, list)) else output
                    if torch.is_tensor(vout):
                        if vout.dim() == 5:
                            vout_slice = vout[:, 0:1, 0:1, 0:1, :]
                        elif vout.dim() >= 3:
                            vout_slice = vout[:, 0:1, :]
                        else:
                            vout_slice = vout
                        if "output" not in all_activations["projector_output"]:
                            all_activations["projector_output"]["output"] = vout_slice.detach()
                            captured_flags["projector_output"].add(key)
                except Exception:
                    pass

            try:
                hooks.append(model.multi_modal_projector.register_forward_hook(projector_forward_hook))
            except Exception as e:
                print(f"Projector hook setup skipped: {e}")

            output = model.generate(**inputs, max_new_tokens=30)
            print(f"Generated tokens: {output}")
            
            # Remove all hooks after generation
            for hook in hooks:
                hook.remove()
            
            # Remove the prompt tokens from the output
            new_tokens = output[0][prompt_length:]
            result = processor.decode(new_tokens, skip_special_tokens=True)

            print(f"Generated result: {result}")
            # Save activations to RDS6
            activation_file = f"{custom_id}_activations.pt"
            activation_path = os.path.join(rds6_checkpoints_dir, activation_file)
            
            # Convert activations to a format that can be saved
            # Move to CPU only when saving to avoid memory issues
            activations_to_save = {}
            for category, layers in all_activations.items():
                activations_to_save[category] = {}
                for layer_name, activation in layers.items():
                    if activation is not None:
                        # Move to CPU and cast to float32 before NumPy conversion (NumPy lacks bfloat16 support)
                        activation_cpu = activation.detach().to(dtype=torch.float32, device="cpu").numpy()
                        activations_to_save[category][layer_name] = activation_cpu
                        # Clear GPU memory for this activation
                        del activation
                    else:
                        activations_to_save[category][layer_name] = None
                
                # Clear GPU memory for this category
                all_activations[category].clear()
            
            # Save activations
            torch.save(activations_to_save, activation_path)
            print(f"Saved activations to RDS6: {activation_path}")
            print(f"Captured activations from 3 strategic layers: {selected_layers}")
            
            # Also save a summary of activation statistics computed from saved activations
            activation_summary = {
                "custom_id": custom_id,
                "prompt_length": prompt_length,
                "generated_length": len(new_tokens),
                "activation_stats": {}
            }
            for category, layers in activations_to_save.items():
                activation_summary["activation_stats"][category] = {}
                for layer_name, activation_np in layers.items():
                    if activation_np is not None:
                        activation_summary["activation_stats"][category][layer_name] = {
                            "shape": list(activation_np.shape),
                            "mean": float(np.mean(activation_np)),
                            "std": float(np.std(activation_np)),
                            "min": float(np.min(activation_np)),
                            "max": float(np.max(activation_np))
                        }
                    else:
                        activation_summary["activation_stats"][category][layer_name] = None

            # Save activation summary
            summary_file = f"{custom_id}_activation_summary.json"
            summary_path = os.path.join(rds6_checkpoints_dir, summary_file)
            with open(summary_path, 'w') as f:
                json.dump(activation_summary, f, indent=2)

            # Clear GPU memory and optimize for next iteration
            del activations_to_save
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create output dictionary
            out_dict = {
                "custom_id": custom_id,
                "content": result,
                "activation_file": activation_file,
                "summary_file": summary_file
            }
            
            # Write to output file
            out_f.write(json.dumps(out_dict) + '\n')
            
            print(f"Processed {i + 1} entries")
    
    print(f"Processing complete. Results written to {output_path}")
    print(f"Activations saved to RDS6: {rds6_checkpoints_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True,
                        help="Directory containing images and batch file")
    parser.add_argument("-m", "--model", choices={"llama11B", "llama90B"}, required=True,
                        help="Model to use for inference")
    parser.add_argument("-b", "--batchFile", default="batch_requests_llamaLocal_0.jsonl",
                        help="Name of the batch file with requests")
    parser.add_argument("-o", "--outputFile", default="Responses.jsonl",
                        help="Name of the output file for responses")
    args = parser.parse_args()
    load_model_and_run(args.directory, args.model, args.batchFile, args.outputFile)
