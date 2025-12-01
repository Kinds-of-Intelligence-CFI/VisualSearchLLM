#!/bin/bash

# Usage: ./plot_results.sh <experiment>
# Experiments: 2Among5, LightPriors, CircleSizes

EXPERIMENT=$1

if [ -z "$EXPERIMENT" ]; then
    echo "Usage: $0 <experiment>"
    echo "Experiments: 2Among5, LightPriors, CircleSizes"
    exit 1
fi

if [ "$EXPERIMENT" == "2Among5" ]; then
    python3.13 analyseResults.py -m cell -e 2Among5 \
        -d 2Among5-Dis-Cells 2Among5-ShapeColourConj-Cells 2Among5-ShapeConj-Cells 5Among2-Dis-Cells 5Among2-ShapeConj-Cells \
        -l "2Among5 Disjunctive" "2Among5 Shape-Colour Conjunctive" "2Among5 Shape Conjunctive" "5Among2 Disjunctive" "5Among2 Shape Conjunctive" \
        -g "Disjunctive:2Among5-Dis-Cells,5Among2-Dis-Cells" "Shape Conjunctive:2Among5-ShapeConj-Cells,5Among2-ShapeConj-Cells" "Shape-Colour Conjunctive:2Among5-ShapeColourConj-Cells" \
        --save --output_dir results/plots/2Among5

elif [ "$EXPERIMENT" == "LightPriors" ]; then
    python3.13 analyseResults.py -m cell -e LightPriors \
        -d LitSpheresTop LitSpheresBottom LitSpheresLeft LitSpheresRight \
        -l "Light Priors Top" "Light Priors Bottom" "Light Priors Left" "Light Priors Right" \
        --save --output_dir results/plots/LightPriors

elif [ "$EXPERIMENT" == "CircleSizes" ]; then
    python3.13 analyseResults.py -m cell -e CircleSizes \
        -d CircleSizesSmall CircleSizesMedium CircleSizesLarge \
        -l "Circle Sizes Small" "Circle Sizes Medium" "Circle Sizes Large" \
        --save --output_dir results/plots/CircleSizes

else
    echo "Unknown experiment: $EXPERIMENT"
    exit 1
fi
