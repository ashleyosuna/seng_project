# seng_project

## How to train LSTM model

The data embedding for the LSTM model uses GloVe, which contains global vector representations of words. You will need to download the zip containing the vectors from their website. Make sure to `cd` into the root directory of this repo before executing commands and scripts.

For linux:

```console
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

For windows

```console
curl -O http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

Next, you will need to get the Reddit post data for preprocessing. This will ideally be done once and then committed, but if changes are made to the api.py script, it may need to be done again. Just run the `src/api.py` script - it may take a while depending on the amount of post data. The data will be put into `post_data.csv` in the root directory.

Finally, you are ready to run the `LSTM.py` script to train the model.
