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


## Additional details and project components

For convenience, the data was pre-downloaded and pushed into the repository. The dataset can be found at `https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews`. Finally due to extremely long processing times, certain data processing steps were saved such as data augmentation. Further explanation will be provided in the report to explain how these files were created.
 

The log folder corresponds to the results of the callbacks during training iterations for each model. The latter can be easily visualised through the command line by running `tensorboard --logdir { path of the cloned repo }/AMLSII_final_assignment/logs`.
 

 The functions.py folder was used to gather all the functions used for data pre-processing, cleaning and transformation.Functions were commented in detail for better readability.Please not that thre functions are defined at the begining of the main.ipynb file as they required some of the local variables returned.In addition the functions defining the different models were kept in that file too in order to allow users to understand better model composition.
 

The file `tokenisation.py` was downloaded from the official tensorflow github as a solution to solve package installation errors. The file was pre-downloaded for your convenience by executing the following command through the terminal `!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py`.
 

