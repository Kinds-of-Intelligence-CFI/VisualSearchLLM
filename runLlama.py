import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import json
import os

def load_model_and_run(directory, model_choice, batch_file, output_file):
    # Select the appropriate model based on the choice
    if model_choice == "llama11B":
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    elif model_choice == "llama90B":
        model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"
    else:
        raise ValueError(f"Unsupported model choice: {model_choice}")
    
    print(f"Loading model: {model_id}")
    actualDirectory="results/"+directory
    # Load model and processor
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Construct full file paths
    input_path = os.path.join(actualDirectory, batch_file)
    output_path = os.path.join(actualDirectory, output_file)
    
    print(f"Processing batch file: {input_path}")
    print(f"Writing results to: {output_path}")
    
    # Process each line in the input file
    with open(input_path, 'r') as f, open(output_path, 'w') as out_f:
        for i, line in enumerate(f):
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
            
            # Process the input
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image, 
                input_text, 
                add_special_tokens=False, 
                return_tensors="pt"
            ).to(model.device)
            

            prompt_length = inputs["input_ids"].shape[-1]

            # Generate the output
            output = model.generate(**inputs, max_new_tokens=30)
            
            # Remove the prompt tokens from the output
            new_tokens = output[0][prompt_length:]
            result = processor.decode(new_tokens, skip_special_tokens=True)




            # Create output dictionary
            out_dict = {
                "custom_id": custom_id,
                "content": result
            }
            
            # Write to output file
            out_f.write(json.dumps(out_dict) + '\n')
            
            
            print(f"Processed {i + 1} entries")
    
    print(f"Processing complete. Results written to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True,
                        help="Directory containing images and batch file")
    parser.add_argument("-m", "--model", choices={"llama11B", "llama90B"}, required=True,
                        help="Model to use for inference")
    parser.add_argument("-b", "--batchFile", default="batch_requests_llamaLocal_0.jsonl",
                        help="Name of the batch file with requests")
    parser.add_argument("-o", "--outputFile", default="responses.jsonl",
                        help="Name of the output file for responses")
    args = parser.parse_args()
    load_model_and_run(args.directory, args.model, args.batchFile, args.model+args.outputFile)