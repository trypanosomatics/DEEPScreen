import os
import sys
import cv2
import json
import torch
import random
import warnings
import subprocess
import numpy as np
import pandas as pd
import torch.nn as nn
from models import CNNModel1
from torch.utils.data import Dataset
from logging_deepscreen import logger
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from data_processing import save_comp_imgs_from_smiles

def get_device():
    if torch.cuda.is_available():
        return 'gpu'
    else:
        return 'cpu'

class DEEPScreenDatasetPredict(Dataset):
    def __init__(self, molecules_to_predict_abs_path:str):
            self.path = molecules_to_predict_abs_path
            self.molecules_files = os.listdir(self.path)

            for file in self.molecules_files:
                if file.find('.png') == -1:
                    logger.debug(f"{file} isn't a .png and cannot be predicted")
                    self.molecules_files.remove(file)

            try:
                assert len(self.molecules_files) > 0
            except:
                logger.error('No files to predict')
                raise RuntimeError('No files to predict')

            logger.debug('Prediction dataset created')

    def __len__(self):
        return len(self.molecules_files)

    def __getitem__(self, index):
        comp_id = self.molecules_files[index]
        comp_id_no_ext = comp_id[:comp_id.index('.png')]
        img_path = os.path.join(self.path, comp_id)
        img_arr = cv2.imread(img_path)
        img_arr = np.array(img_arr) / 255.0

        # img_arr = img_arr.transpose((2, 0, 1))
        return img_arr, comp_id_no_ext


def loader_generator(path_to_molectules_to_predict:str)->torch.utils.data.DataLoader:
    dataset = DEEPScreenDatasetPredict(path_to_molectules_to_predict)
    return torch.utils.data.DataLoader(dataset)


def predict(trained_model_path:str, path_to_molectules_to_predict:str, fully_layer_1, fully_layer_2, drop_rate )->dict:
    device = get_device()

    data_loader = loader_generator(path_to_molectules_to_predict)

    # loading model
    model = CNNModel1(fully_layer_1, fully_layer_2, drop_rate).to()
    model.load_state_dict(trained_model_path)
    model.eval()

    prediction = dict()
    with torch.no_grad():
        for data in data_loader:
            img_arrs, comp_ids = data
            img_arrs = torch.tensor(img_arrs).type(torch.FloatTensor).to(device)
            y_pred = model(img_arrs).to(device)
            _, preds = torch.max(y_pred, 1)
            prediction[comp_ids[0]] = preds

    return prediction

def smiles_to_img_png(comp_id, smiles, output_path, IMG_SIZE:int=200):
    mol = Chem.MolFromSmiles(smiles)
    DrawingOptions.atomLabelFontSize = 55
    DrawingOptions.dotsPerAngstrom = 100
    DrawingOptions.bondLineWidth = 1.5
    Draw.MolToFile(mol, os.path.join(output_path, f"{comp_id}.png"), size= (IMG_SIZE, IMG_SIZE))

