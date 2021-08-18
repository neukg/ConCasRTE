## CIKM 2021: A Conditional Cascade Model for Relational Triple Extraction.

## Requirements
The main requirements are:
- python 3.6
- torch 1.4.0 
- tqdm
- transformers == 2.8.0

## Usage
* **Get pre-trained BERT model**

  Download [BERT-BASE-CASED](https://huggingface.co/bert-base-cased) and put it under `./pretrained`.

* **Train and select the model**
```
python run.py --dataset=NYT   --num_train_epochs=100 --batch_size=18 --train

python run.py --dataset=WebNLG  --num_train_epochs=50 --batch_size=6  --train

python run.py --dataset=NYT_simple  --num_train_epochs=100  --batch_size=18 --train

python run.py --dataset=WebNLG_simple --num_train_epochs=50 --batch_size=6 --train
```

* **Evaluate on the test set**

```
python run.py --dataset NYT

python run.py --dataset WebNLG

python run.py --dataset NYT_simple

python run.py --dataset WebNLG_simple
```

### Acknowledgement
Parts of our codes come from [bert4keras](https://github.com/bojone/bert4keras).
