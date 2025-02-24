import json
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor



def load_model_and_run(directory):

    # Load the model and processor
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    batch_file = "results/"+directory+"/batch_requests_llama_0.jsonl"
    output_file = "results/"+directory+"/llama_responses.jsonl"

    batch_data = []
    with open(batch_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                entry = json.loads(line)
                batch_data.append(entry)

    # Open the output file in write mode
    with open(output_file, "w") as out_f:
        # Iterate over each entry in the jsonl file
        for entry in batch_data:
            custom_id = entry["custom_id"]
            messages = entry["messages"]

            # Load the corresponding image
            image_path = f"results/{directory}/{custom_id}"
            image = Image.open(image_path)

            # Prepare the input text using the provided messages
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)

            # Generate the model output
            output = model.generate(**inputs, max_new_tokens=30)

            # Decode the generated response
            prompt_length = inputs['input_ids'].shape[-1]
            new_tokens = output[0][prompt_length:-1]
            result = processor.decode(new_tokens)
            
            # Create an output dictionary for this entry
            out_dict = {
                "custom_id": custom_id,
                "content": result.strip()
            }
            # Write the dictionary as a JSON line to the output file
            out_f.write(json.dumps(out_dict) + "\n")
            print(f"Logged result for {custom_id}: {result.strip()}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True)
    args = parser.parse_args()
    load_model_and_run(args.directory)

