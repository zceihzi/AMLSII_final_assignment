<!-- # AMLSII_final_assignment
pip install progressbar
pip install contractions
pip install sentencepiece

pip install spacy
pip install gensim 
pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html
pip install fastBPE regex requests sacremoses subword_nmt
pip install -q hydra-core
pip install tqdm

!pip install textattack -q
!pip install GPUtil

python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
python -m textblob.download_corpora
wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py -->

<!--- These are examples. See https://shields.io for others or to customize this set of shields. You might want to include dependencies, project status and licence info here --->

# AMLSII_final_assignment

AMLSII_final_assignment is a `Swntiment classifier` written in `Python 3.8` to  solve the task of `sentiment analysis` using tensorflow and Keras .
The aim of this project is to provide an understanding of the diffenrent methods that could be used to classify text data. Classification involves a fine grained analysis of the corpus where sentiments are labels as: 
* Negative
* Somewhat negative
* Neutral
* Somewhat positive 
* Positive

## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have installed a Python version not older that `Python3.5`

## Cloning the repo and installing dependencies


