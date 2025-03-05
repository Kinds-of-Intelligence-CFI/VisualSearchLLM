import json
import base64
from together import Together

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_model_and_run(directory, model, batchFile, outputFile):
    # Initialize Together AI client
    client = Together()

    modelMap = {"llama11B": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", 
                "llama90B": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"}
    modelString = modelMap[args.model]

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
            prompt_parts = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(str(item) for item in content)
                else:
                    content = str(content)
                prompt_parts.append(content)
            prompt_text = " ".join(prompt_parts)
            #print(prompt_text)
            # Determine the image path and encode the image.
            image_path = f"results/{directory}/{custom_id}"
            base64_image = encode_image(image_path)
            image_data_url = f"data:image/jpeg;base64,{base64_image}"

            # Call Together AI with a message containing both text and image.
            response = client.chat.completions.create(
                model=modelString,
                temperature=0.0,
                top_k=50,
                messages=messages,
            )

            # Get the model's output text.
            result = response.choices[0].message.content.strip()

            # Prepare and write the output.
            out_dict = {
                "custom_id": custom_id,
                "content": result
            }
            out_f.write(json.dumps(out_dict) + "\n")
            print(f"Logged result for {custom_id}: {result}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True)
 
    parser.add_argument("-m", "--model", choices={"llama11B", "llama90B"}, required=True)
    parser.add_argument("-b", "--batchFile", default="batch_requests_0.jsonl")
    parser.add_argument("-o", "--outputFile", default="responses.jsonl")
    args = parser.parse_args()
    load_model_and_run(args.directory, args.model, args.batchFile, args.outputFile)
