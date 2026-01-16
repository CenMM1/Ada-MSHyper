# Preprocess Directory

This directory contains scripts for preprocessing multimodal datasets used in the Ada-MSHyper project.

## download_mosei.sh

download the dataset from onedrvie, replace the link and the file name for your own use.

because i forget to add the label.csv into the zip file, you need to put them in to the dataset folder your unziped. and rename it to label.csv

## CHERMA dataset

chema.py

This script is used to divide the CHERMA dataset.

This will load up to 5000 samples, split them into train/dev/test sets, and save them as PyTorch tensors in the `./processed_data/` directory.

## MOSI & MOSEI dataset

msa_fet.py is for audio and text, run this one first to retrive the features, change the path befor run it.

multimodal_data.py is for video, and combine the audio and text features, then generate the train val and test.

### Envoriment:
conda create -n py310 python=3.10
conda activate py310
pip install MMSA-FET
pip install "numpy<2"
python -m MSA_FET install
pip install facenet-pytorch torch torchvision tqdm
pip install opencv-python==4.11.0.86


