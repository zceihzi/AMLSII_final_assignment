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

AMLSII_final_assignment is a `Sentiment classifier` written in `Python 3.8` to  solve the task of `sentiment analysis` mainly using Tensorflow and Keras.The aim of this project is to provide an understanding of the diffenrent methods that could be used to classify text data. Classification involves a fine grained analysis of the corpus where sentiments are labels as: 
* Negative
* Somewhat negative
* Neutral
* Somewhat positive 
* Positive

## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have installed a Python version not older that `Python3.5`

## Cloning the repo 
* This repository can be easily cloned using teh command `git clone https://github.com/zceihzi/AMLSII_final_assignment.git`
* The file main.ipynb contains the main logic of the code and will be described in detail in the report

## Software advice and package settings 

Since the code was submitted in a notebook format, it is possible to visualise the output of each cell by simply clicky on the file directly on this page. If any user wishes to test the code, it is recommended to run the main.ipynb through a Jupiter notebook, or alternatively on Visual studio code. Please note that these are only suggestions. To run this file, basic python libaries need to be installed such as Pandas, numpy and so on.\
 \
The code also uses other less common packages such as spacy, gensim or textattack. Thus, please make sure that the following libraries are installed in your machine. Otherwise please use the guidelines below to install thenm through your terminal: 

* `progressbar` (pip install progressbar)
* `contractions` (pip install contractions)
* `spacy` (pip install spacy)
* `gensim` (pip install gensim )
* `torch` (pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html)
* `fastBPE` (pip install fastBPE regex requests sacremoses subword_nmt)
* `hydra-core` (pip install -q hydra-core)
* `textattack` (pip install textattack -q)
* `tqdm` (pip install tqdm)
* `tqdm` (pip install tqdm)
* `tqdm` (pip install tqdm)


For spacy depency files please execute: 

* python -m spacy download en_core_web_sm
* python -m spacy download en_core_web_lg
* python -m textblob.download_corpora


## Additional details
The file `tokenisation.py` was downloaded from the official tensorflow github as a solution to solve package installation errors. The file was obtained by executing the following command through the terminal `!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py`.\
 \
For convenience, the data was pre-downloaded and pushed into the repository. The dataset can be found at `https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews`. Finally due to extremely long processing times, certain data processing steps were saved such as data augmentation. Further explanation will be provided in the report to explain how these files were created.

