# NestedNER

## Prerequisites
The main requirements are:
* tqdm
* flair
* word2vec
* glove-python-binary==0.2.0
* transformers==2.10.0
* wandb # for logging the results
* yaml

In the root directory, run
```bash
pip install -e .
```
## Data Format
```
{
  "id": "train_2975", 
  "text": "ll Services Plan Documents Only ( Section 125 ) Premium Offset Plan ( POP ) Wellness Plans MBTA Corporate Transit Pass Program WMATA Smar Trip Transit / Parking Program Contact Us HRC Total Solutions 111 Charles Street Manchester , New hampshire 03101 customerservice @ hrcts . com Phone : ( 603 ) 647 - 1147 Fax : ( 866 ) 978 - 7868 Follow Us Linked In You Tube Twitter Resources IIAS Participating Pharmacies IIAS 90 % Merchant List FSA Store HSA St", 
  "entity_list": [
      {"text": "111 Charles Street", "type": "detail", "char_span": [200, 218]}, 
      {"text": "Manchester", "type": "city", "char_span": [219, 229]}, 
      {"text": "New hampshire", "type": "state", "char_span": [232, 245]}, 
      {"text": "03101", "type": "zipcode", "char_span": [246, 251]},
  ]
}
```

## Pretrained Model and Word Embeddings
Download [BERT-BASE-CASED](https://huggingface.co/bert-base-cased) and put it under `../pretrained_models`. Download [word embeddings](https://pan.baidu.com/s/1HcigyVBjsShTR2OAEzj5Og) (code: 8044) and put them under `../pretrained_emb`.

## Train
Set configuration in `tplinker_ner/train_config.yaml`. Start training:
```
cd tplinker_ner
python train.py
```

## Evaluation
Set configuration in `tplinker_ner/eval_config.yaml`. Start evaluation by running `tplinker_ner/Evaluation.ipynb`

