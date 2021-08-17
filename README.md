
## CIKM 2021: A Conditional Cascade Model for Relational Triple Extraction.

## Requirements
The main requirements are:
- python 3.6
- torch 1.4.0 
- tqdm
- transformers == 2.8.0

## Usage
1. **Train and select the model**

python run.py --dataset=NYT  --conf_value=0.1 --train

python run.py --dataset=WebNLG  --conf_value=0.3 --train

2. **Evaluate on the test set**

python run.py --dataset NYT

python run.py --dataset WebNLG
