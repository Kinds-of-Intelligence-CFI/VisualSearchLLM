
# VisualSearchLLM


This code is for investigating the pop-out effect in Visual AI systems.

To run this code from scratch to replicate our results the following should be done:

## Install requirements

- Create a python environment `python -m venv visualSearch`
- Activate the environment (shown for Windows) `.\visualSearch\Scripts\Activate.ps1`
- Install the requirements `pip install -r requirements.txt`

## Generate Images

The dataset of images with varying visual processing demands.
The datasets can be created using the generateImages.py file. 
The intended use is `python generateImages.py -n x -d x -p x`
where `-n` is the flag for the number of images to create, `-d` is the directory to store the images, and `-p` is a selection from preset options configuring the images created. More control over the images generated is possible using other flags.

Our results were created with n=10000 and we used the following presets:

| Experiment Type     | Presets to Use                          |
|---------------------|------------------------------------------|
| 2Among5       | `2Among5ColourRand`<br>`2Among5NoColourRand`<br>`2Among5ConjRand`<br> `5Among2ColourRand`<br>`5Among2NoColourRand`<br>`5Among2ConjRand` |
| Light Priors  | `VerticalGradient`<br>`VerticalGradientReversed`<br>`HorizontalGradient`<br>`HorizontalGradientReversed` |
| Circle Sizes  | `CircleSizesSmall`<br>`CircleSizesMedium`<br>`CircleSizesLarge` |


## Create a Batch

For OpenAI and Anthropic models we can take advantage of Batching to reduce costs. To do this we need to prepare all of the requests ahead of time.
We can do this with `python createBatch.py -d x -m x -p x`
where the `-d` flag gives the directory; `-m` the model (see supported model list), `-p` gives the prompt to provide as an element from the list in `constructMessage.py`
This creates an appropriate number of `.jsonl` files for the batching (there is a maximum number of requests per batch we frequently exceed).

Our results were created with the following prompt presets (found in `constructMessage.py`)

| Cells | Presets to Use                          |
|---------------------|------------------------------------------|
| (2Among5) Inefficient Disjunctive <br> Efficient Disjunctive     | `std2x2-2Among5`<br> `std2x2-5Among2` |
| (2Among5) Conjunctive | `std2x2-2Among5-conj` |
| Light Priors  | `lightPriorsHorizontal`<br>`lightPriorsVertical`<br>`lightPriorsHorizontalReversed`<br>`lightPriorsVerticalReversed` |
| Circle Sizes  | `circle-sizes` |



| Coordinates | Presets to Use                          |
|---------------------|------------------------------------------|
| (2Among5) Inefficient Disjunctive <br> Efficient Disjunctive     | `coords-2Among5`<br> `coords-5Among2` |
| (2Among5) Conjunctive | `coords-2Among5-conj` |
| Light Priors  | `coords-lightPriorsHorizontal`<br>`coords-lightPriorsVertical`<br>`coords-lightPriorsHorizontalReversed`<br>`coords-lightPriorsVerticalReversed`|
| Circle Sizes  | `coords-circle-sizes` |



## Submit a Batch
To actually submit the batch to Anthropic/OpenAI:
`python submitBatch.py -d x -m x`
This will handle the creation of a job with the batching provider and save the appropriate log numbers for retrieval later.
Similarly, for Llama, `python submitHPCBatch.py -d x -m x` will submit the batch to SLURM.


## Checking on / Finishing a Batch
For the Anthropic/OpenAI models, batches occur asynchronously, seemingly whenever the providers have available compute. To check the progress of a batch use
`python checkBatchProgress.py -d x -m x`
This will display, for each batch, the number of requests completed. Once all batches are complete, the results files will be downloaded.
If running a request locally or on an HPC you'll need to look up how to check when your batch is done yourself.

## Processing Batch Results
To turn the batch results into a more readable format we need to pocess them. To do this we use
`python processBatchResults.py -d x -m x` 
We have a few extra options depending on whether we are expecting the results to be coordinates or cells.
Either `-c`, `-rc`.
The script `processAllJobs.py` can let you select models to process batch results for multiple models in sequence.



## Presenting the Results
`analyseResults.py` is the main script for visualising results. Usage looks like
`python analyseResults.py -d <d1, d2, d3,d4...>, <l1,l2,l3,l4...> -g group1:d1,d2 group2:d3,d4 --mode m`
where `<d1,...>` is list of directories which results you want to compare, `<l1,..>` is a list of labels, and `groupx: d1,d2` names a new group, groupx and merges d1 and d2's results into a single blob. 
This is useful e.g., when wanting to merge 2Among5 and 5Among2.