#!/bin/bash

source .env

python train_classifieur.py\
 --logdir ./expe_log/\
 --datadir ./"$DATA_DIR"/\
 --csv ./"$DATA_DIR"/metadata.csv\
 --weights_col WEIGHTS\
 --csv_out ./expe_log/preds.csv