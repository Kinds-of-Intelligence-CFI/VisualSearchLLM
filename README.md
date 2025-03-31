# VisualSearchLLM


This code is for investigating the pop-out effect in Visual AI systems.

To run this code from scratch the following should be done:

## Install requirements

- Create a python environment `python -m venv visualSearch`
- Activate the environment (shown for Windows) `.\visualSearch\Scripts\Activate.ps1`
- Install the requirements `pip install -r requirements.txt`

## Generate Images

The dataset of images with varying visual processing demands.
The datasets can be created using the generateImages.py file. 
The intended use is `python generateImages.py -n x -d x -p x`
where `-n` is the flag for the number of images to create, `-d` is the directory to store the images, and `-p` is a selection from preset options configuring the images created. More control over the images generated is possible using other flags.

## Create a Batch

For OpenAI and Anthropic models we can take advantage of Batching to reduce costs. To do this we need to prepare all of the requests ahead of time.
We can do this with `python createBatch.py -d x -m x -p x`
where the `-d` flag gives the directory; `-m` the model (GPT-4o or Claude Sonnet), `-p` gives the prompt to provide as an element from the list in `constructMessage.py`
This creates an appropriate number of `.jsonl` files for the batching (there is a maximum number of requests per batch we frequently exceed).

## Submit a Batch
To actually submit the batch to Anthropic/OpenAI:
`python submitBatch.py -d x -m x`
This will handle the creation of a job with the batching provider and save the appropriate log numbers for retrieval later.

## Checking on / Finishing a Batch
Batches occur asynchronously, seemingly whenever the providers have available compute. To check the progress of a batch use
`python checkBatchProgress.py -d x -m x`
This will display, for each batch, the number of requests completed. Once all batches are complete, the results files will be downloaded.

## Processing Batch Results
To turn the batch results into a more readable format we need to process them. To do this we use
`python processBatchResults.py -d x -m x` 
We have a few extra options depending on whether we are expecting the results to be coordinates, cells, or quadrant numbers. 
Either `-c`, `-rc`, `-q`

## Presenting the Results
To generate visualisations of the results there are three options, depending on whether the questions formed around coordinates, cells, or quadrant numbers.
The first is to 

If expecting coordinates, `python coordsAnalysis -d x -l x`,
where `-d` is a sequence of directories with completed results files, and `-l` a sequence of corresponding labels.
Similarly if expecting cells, use `python cellsAnalysis -d x -l x`
Finally, if expecting quadrant numbers use `python resultsAnalysis -d x -l x`