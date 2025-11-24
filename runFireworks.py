import json
import base64
import os
from openai import OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_model_and_run(directory, model, batchFile, outputFile):
    # Initialize OpenAI client for Fireworks AI
    client = OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=os.environ.get("FIREWORKS_API_KEY"),
    )

    # Map model choices to Fireworks AI model IDs
    # Note: These are standard Fireworks model IDs. Update if using a custom deployment.
    modelMap = {
        "llama11B": "accounts/fireworks/models/llama-v3p2-11b-vision-instruct", 
        "llama90B": "accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
        "Qwen7B": "accounts/johnburden/deployedModels/qwen2p5-vl-7b-instruct-bhcf0qib",
        "Qwen32B": "accounts/johnburden/deployedModels/qwen2p5-vl-32b-instruct-ghx8oltt",
        "Qwen72B": "PLACEHOLDER_FOR_72B_MODEL_PATH"
    }
    
    
    if model not in modelMap:
        raise ValueError(f"Model {model} not found in modelMap. Available: {list(modelMap.keys())}")
        
    modelString = modelMap[model]
    print(f"Using model: {modelString}")

    batch_file = f"results/{directory}/{batchFile}"
    output_file = f"results/{directory}/{outputFile}"
    
    batch_data = []
    with open(batch_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                entry = json.loads(line)
                batch_data.append(entry)
                
    with open(output_file, "w") as out_f:
        for entry in batch_data:
            custom_id = entry["custom_id"]
            messages = entry["messages"]

            # Build a prompt text handling content that might be a list or string.
            # This part mimics runTogether.py logic for prompt construction if needed,
            # but for Chat Completions we usually pass messages directly.
            # However, we need to handle the image encoding and insertion into the message.
            
            # The input messages in batch_requests usually have a placeholder or just text.
            # We need to inject the image data URL.
            
            # Determine the image path and encode the image.
            image_path = f"results/{directory}/{custom_id}"
            if not os.path.exists(image_path):
                 print(f"Warning: Image not found at {image_path}, skipping image attachment.")
                 base64_image = None
            else:
                base64_image = encode_image(image_path)
            
            # Reconstruct messages to include the image url if not present
            # Assuming the last message is the user message where we want to attach the image
            # or the messages structure already expects an image_url but we need to fill it?
            # runTogether.py does:
            # image_data_url = f"data:image/jpeg;base64,{base64_image}"
            # response = client.chat.completions.create(..., messages=messages)
            # Wait, runTogether.py calculates image_data_url but DOES NOT USE IT in the messages passed to client?
            # Let's check runTogether.py again.
            
            # In runTogether.py:
            # image_path = f"results/{directory}/{custom_id}"
            # base64_image = encode_image(image_path)
            # image_data_url = f"data:image/jpeg;base64,{base64_image}"
            # response = client.chat.completions.create(..., messages=messages)
            
            # It seems runTogether.py assumes 'messages' already contains the image info OR it's missing something?
            # Ah, looking at runTogether.py again (lines 45-55):
            # It calculates image_data_url but doesn't seem to insert it into 'messages'.
            # Unless 'messages' in the batch file already has the base64 string?
            # Or maybe I missed where it modifies 'messages'.
            # Let's re-read runTogether.py carefully.
            
            # Lines 33-42 build prompt_text but don't use it.
            # Lines 45-47 build image_data_url but don't use it.
            # Line 54 passes 'messages' directly.
            
            # This implies the batch file might already have the image or runTogether.py is incomplete/buggy?
            # OR, maybe the 'messages' in the batch file are just text and the model is text-only?
            # But it uses "Llama-3.2-11B-Vision-Instruct-Turbo".
            # If I look at runLlama.py, it loads the image from disk.
            
            # I should probably fix this or at least do what's right for Fireworks.
            # For Fireworks/OpenAI format, we need to pass the image in the content list.
            
            new_messages = []
            for msg in messages:
                new_content = []
                if msg["role"] == "user":
                    # Check if content is string or list
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        new_content.append({"type": "text", "text": content})
                    elif isinstance(content, list):
                        new_content.extend(content)
                    
                    # Attach image if it's the user message (usually the last one or the one with text)
                    # We'll attach it to the first user message we find or all? usually just one image per request in this context?
                    # Let's assume we append the image to the user message.
                    if base64_image:
                        image_data_url = f"data:image/jpeg;base64,{base64_image}"
                        # Check if image is already in content
                        has_image = any(item.get("type") == "image_url" for item in new_content)
                        if not has_image:
                            new_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url
                                }
                            })
                    
                    new_messages.append({"role": "user", "content": new_content})
                else:
                    new_messages.append(msg)

            # Call Fireworks AI
            try:
                response = client.chat.completions.create(
                    model=modelString,
                    temperature=0.0,
                    messages=new_messages,
                    extra_body={
                        "top_k": 50
                    }
                )

                # Get the model's output text.
                result = response.choices[0].message.content.strip()
                print(f"Logged result for {custom_id}: {result}")

            except Exception as e:
                print(f"Error processing {custom_id}: {e}")
                result = f"ERROR: {e}"

            # Prepare and write the output.
            out_dict = {
                "custom_id": custom_id,
                "content": result
            }
            out_f.write(json.dumps(out_dict) + "\n")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True)
    parser.add_argument("-m", "--model", choices={"llama11B", "llama90B", "Qwen7B", "Qwen32B", "Qwen72B"}, required=True)
    parser.add_argument("-b", "--batchFile", default="batch_requests_0.jsonl")
    parser.add_argument("-o", "--outputFile", default="responses.jsonl")
    args = parser.parse_args()
    
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("Error: FIREWORKS_API_KEY environment variable not set.")
        exit(1)
        
    load_model_and_run(args.directory, args.model, args.batchFile, args.outputFile)
