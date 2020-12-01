# Long Document Coreference with Bounded Memory Neural Networks
Code for the EMNLP 2020 paper [Learning to Ignore: Long Document Coreference with Bounded Memory Neural Networks](https://arxiv.org/pdf/2010.02807.pdf)

## Environment Setup

### Install  Requirements
The codebase has been tested for:
```
python >= 3.7.0
torch==1.6.0
transformers==3.0.0
```
These can be separately installed or just do:
```
pip install -r requirements
```

### Clone a few Github Repos
```
# Clone this Repo
git clone https://github.com/shtoshni92/long-doc-coref
cd long-doc-coref/

# Clone Reference Scorers
git clone https://github.com/conll/reference-coreference-scorers

# Clone LitBank Coreference Repo
git clone https://github.com/dbamman/lrec2020-coref
```



## Data
Step 1: Get CoNLL format data
* LitBank - Get the 10-fold cross-validation data from [here](https://github.com/dbamman/lrec2020-coref/tree/master/data). 
Need to run the : 
```
python scripts/create_crossval.py data/litbank_tenfold_splits data/original/conll/  data/litbank_tenfold_splits
```
* OntoNotes - Follow the recipe from [here](https://github.com/mandarjoshi90/coref/blob/master/setup_training.sh) to process the data. 

Step 2: Chunk documents into BERT segments (max segment length is set to 512 in code). Here are the steps for LitBank:
```
OUTPUT_DIR="../data/litbank"
CONLL_DIR="../data/litbank/

```

* LitBank + Independent: python data_processing/independent_litbank.py $PATH_TO_CONLL_DIR $OUTPUT_DIR
* LitBank + Overlap: python data_processing/independent_litbank.py $PATH_TO_CONLL_DIR $OUTPUT_DIR


## Training/Evaluation

We use a two-stage training wherein we first train mention detection models and then the coreference model. 
As long as both the mention detection models and coreference models are stored in the same directory, 
the coreference model will be able to pick up the relevant pretrained mention detection model. 
_This step might be done away with because for OntoNotes we found no drop to very slight drop in some accidental runs 
where the model was trained completely end-to-end._ 

### Training mention detection models


### Pretrained Models + Colab Inference notebook


### Important Hyperparams


## Additional Code
The ``notebooks`` directory has some additional code. 
* 

## Citation
```
@inproceedings{toshniwal2020bounded,
    title = {{Learning to Ignore: Long Document Coreference with Bounded Memory Neural Networks}},
    author = "Shubham Toshniwal and Sam Wiseman and Allyson Ettinger and Karen Livescu and Kevin Gimpel",
    booktitle = "EMNLP",
    year = "2020",
}
```

