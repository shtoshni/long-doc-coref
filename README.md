# Long Document Coreference with Bounded Memory Neural Networks
Code for the EMNLP 2020 paper [Learning to Ignore: Long Document Coreference with Bounded Memory Neural Networks](https://arxiv.org/pdf/2010.02807.pdf).
Pretrained models are released [here](https://drive.google.com/drive/folders/1UFhkrlBP-O2MeaxVygZcuP9RWuglOTmN?usp=sharing).

[Colab Notebook](https://colab.research.google.com/drive/1aG37Fkgg4GILFvpGE7YALEf-TWJlZgTe?usp=sharing) to show how to perform inference with pretrained models. 


## Environment Setup

### Install  Requirements
The codebase has been tested for:
```
python==3.6
torch==1.6.0
transformers==4.2.2
scipy==1.4.1
```
These are the core requirements which can be separately installed or just do:
```
pip install -r requirements.txt
```

Clone a few Github Repos (including this!)
```
# Clone this repo
git clone https://github.com/shtoshni92/long-doc-coref
cd long-doc-coref/
mkdir -p resources; cd resources/

# Clone Official Scorers for Coreference
git clone https://github.com/conll/reference-coreference-scorers

# Clone LitBank Coreference Repo
git clone https://github.com/dbamman/lrec2020-coref
```


## Data

### Step 1:  Getting CoNLL files
Process LitBank data to create 10-fold splits
```
cd lrec2020-coref/
python scripts/create_crossval.py data/litbank_tenfold_splits data/original/conll/  data/litbank_tenfold_splits
LITBANK_CONLL_DIR=${PWD}/data/litbank_tenfold_splits/
```

For OntoNotes:
* Download data from [LDC](https://catalog.ldc.upenn.edu/LDC2013T19)
* Follow the relevant portion of the preprocessing recipe from [here](https://github.com/mandarjoshi90/coref/blob/master/setup_training.sh). 


### Step 2: Chunking data into BERT windows (segment length set to 512)
These scripts have been adapted from the [(Span)BERT for Coreference Repo](https://github.com/mandarjoshi90/coref).

First move to `src` root and update `PYTHONPATH`
```
cd ../../src  # Or go to the src/ directory from wherever you're
export PYTHONPATH=${PWD}:$PYTHONPATH
```

Processing LitBank (result of this step available [here](https://drive.google.com/file/d/1LXCLVjjDGjDNAMiKuncEOhTxn-RpSINl/view?usp=sharing))
```
OUTPUT_DIR="../data/litbank"

# Independent windows i.e. no overlap in windows
python data_processing/independent_litbank.py $LITBANK_CONLL_DIR $OUTPUT_DIR/independent

# Overlapping windows
python data_processing/overlap_litbank.py $LITBANK_CONLL_DIR $OUTPUT_DIR/overlap
```

Ontonotes data can be processed in exactly the same way using the equivalent scripts.

## Training/Evaluation

In the paper we use a two-stage training wherein we first train mention detection models and then the coreference model. 
As long as both the mention detection models and coreference models are stored in the same directory, 
the coreference model will be able to pick up the relevant pretrained mention detection model. 
_This step might be done away with because for OntoNotes we found no drop to very slight drop in some accidental runs 
where the model was trained completely end-to-end._ 

Both the stages depend on the pretrained SpanBERT models released [here](https://github.com/facebookresearch/SpanBERT).
These pretrained models include SpanBERT and additional coreference model parameters. 
We have stripped out the additional model parameters, converted them to PyTorch format, and released it via Huggingface modelhub. 
Links for [base model](https://huggingface.co/shtoshni/spanbert_coreference_base) and [large model](https://huggingface.co/shtoshni/spanbert_coreference_large).
_No need to manually do download it, the transformers library will take care of it._ 
 
### Step 1: Training mention detection models
For LitBank we need to train mention detectors for each of the cross validation splits. 

```
# Sweep over cross validation split argument from 0 to 9
python mention_model/main.py -cross_val_split 0 -max_span_width 20 -top_span_ratio 0.3 -max_epochs 25
```
 
For OntoNotes we use the parameters used in the SpanBERT paper (and train for fewer epochs as the dataset is much bigger). 
```
python mention_model/main.py -dataset ontonotes -max_span_width 30 -top_span_ratio 0.4 -max_epochs 10
```
 
### Step 2: Training coreference model
There are a lot of hyperparameters when training these models. I'll go over the few important ones, for the full set 
check out `src/auto_memory_model/main.py`. Also check out `src/auto_memory_model/experiment.py` which has key training and evaluation details.

#### Span options (also used in mention detection)
- `max_span_width`: Used 20 for LitBank and 30 for OntoNotes. 
- `top_span_ratio`: Number of mentions proposed during mention proposal stage divided by document length. 0.3 for LitBank and 0.4 for OntoNotes.
  
#### Memory Type and Number of memory slots 
- `mem_type`: learned -> LB-MEM, lru -> RB-MEM, unbounded -> U-MEM, unbounded_no_ignore -> U-MEM*. Default is "learned".
- `num_cells`: Memory capacity of bounded memory models. Default is 20. Doesn't affect unbounded models.

#### Training Specific Arguments
- `max_training_segments`: Maximum number of BERT windows to which a document is truncated. Documents are truncated by selecting a contiguous sequence of BERT segments. This is key to reducing memory requirements and has been used in previous work. Default is None. We used 3 for OntoNotes and 5 for LitBank.
- `sample_invalid`: Mention proposal stage is supposed to be high recall, and a lot of the mentions proposed in this stage are not part of any coreference clusters (clusters include singletons if singletons are marked). These mentions are called "invalid". During training we can sample a fraction of this which again has memory benefits and sometimes performance gains as well.
- `new_ent_wt`: Weight loss term for not predicting a coref link. Default is 1.0. For LitBank we found gains with increasing this value to 2.0 and 5.0.

For quick debugging, you can use `num_train_docs` to specify the number of training docs. 

#### Evaluation Script Arguments
We use Kenton Lee's implementation of coreference metrics (see `src/coref_utils/metrics.py`). 
For just the final evaluation i.e. when training stops, we used the official CoNLL perl scripts (Github repo cloned earlier).
The code will use the CoNLL evaluation when both the ground truth conll format data and the official conll scripts are available. 
The conll scorere can be specified via `conll_scorer` option.

We assume the ground truth conll format data is available in the same location as the processed data in a `conll` directory. 
So for LitBank if the data with independent segmentation is at `../data/litbank/independent` then the script assumes the conll data is at `../data/litbank/conll`.

A couple of important points:
- For LitBank evaluation, the metrics are reported for the aggregated data across the splits. So the conll ground truth and prediction files need to be aggregated for final evaluation.
- For LitBank I found that the official script can in rare cases break (the python implementation never breaks). Since this happened during hyperparameter search, I just ignored those runs. 
 


#### Sample Runs
Some configs that work for LitBank:
```
# LB-MEM + 20 Cells, Cross validation split 3 - Dev 77.3, Test 76.6
python auto_memory_model/main.py -dataset litbank -mem_type learned -num_cells 20 -top_span_ratio 0.3 -max_span_width 20 -max_epochs 25 -dropout_rate 0.3 -sample_invalid 0.25 -new_ent_wt 2.0 -cross_val_split 3 -max_training_segments 5 -seed 0

# U-MEM, Cross validation split 3 - Dev 77.9, Test 77.0
python auto_memory_model/main.py -dataset litbank -mem_type unbounded -top_span_ratio 0.3 -max_span_width 20 -max_epochs 25 -dropout_rate 0.3 -sample_invalid 0.75 -new_ent_wt 2.0 -cross_val_split 3 -max_training_segments 5 -seed 0
```

Some configs that work for OntoNotes:
```
# LB-MEM + 20 cells - Dev 78.1, Test 78.2
python auto_memory_model/main.py -dataset ontonotes -mem_type learned -num_cells 20 -max_span_width 30 -max_epochs 15 -dropout_rate 0.4 -top_span_ratio 0.4 -sample_invalid 0.5 -label_smoothing_wt 0.1 -max_training_segments 3 -seed 50 

# U-MEM - Dev 78.4, Test 78.1
python auto_memory_model/main.py -dataset ontonotes -mem_type unbounded -max_span_width 30 -max_epochs 15 -dropout_rate 0.4 -top_span_ratio 0.4 -sample_invalid 0.25 -label_smoothing_wt 0.1 -max_training_segments 3 -seed 50 

# U-MEM* - Dev 79.6, Test 79.6
python auto_memory_model/main.py -dataset ontonotes -mem_type unbounded_no_ignore -max_span_width 30 -max_epochs 15 -dropout_rate 0.4 -top_span_ratio 0.4 -sample_invalid 0.75 -label_smoothing_wt 0.01 -max_training_segments 3 -seed 50
```


### Additional Code
The ``notebooks`` directory has some additional code for:
 - Visualizing LitBank data in HTML
 - Aggregating runs on Slurm for choosing hyperparameters
 - Analysis of memory logs

### Citation
```
@inproceedings{toshniwal2020bounded,
    title = {{Learning to Ignore: Long Document Coreference with Bounded Memory Neural Networks}},
    author = "Shubham Toshniwal and Sam Wiseman and Allyson Ettinger and Karen Livescu and Kevin Gimpel",
    booktitle = "EMNLP",
    year = "2020",
}
```

