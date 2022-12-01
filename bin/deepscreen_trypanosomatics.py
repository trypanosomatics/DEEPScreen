import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from models import CNNModel1
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from logging_deepscreen import logger
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolDrawOptions
import random
from train_deepscreen import calculate_val_test_loss
from evaluation_metrics import prec_rec_f1_acc_mcc, get_list_of_scores

RANDOM_STATE = 123

def get_device():
    if torch.cuda.is_available():
        return 'gpu'
    else:
        return 'cpu'

class DEEPScreenDatasetTrain(Dataset):
    def __init__(self, path_tmp_files:str, df_compid_smiles_bioactivity:pd.DataFrame, smiles_column = 'smiles', compound_id_column = 'comp_id', bioactivity_label_column = 'bioactivity_label'):
            self.path = path_tmp_files
            self.path_imgs = os.path.join(self.path,'imgs')
            self.df = df_compid_smiles_bioactivity.copy()
            self.compid_column = compound_id_column
            self.smiles_column = smiles_column
            self.label_column  = bioactivity_label_column
            if not os.path.exists(self.path_imgs):
                os.makedirs(self.path_imgs)

            # creating molecules images -> path will be stored in 'img_molecule' column
            self.df['img_molecule'] = self.df.apply(lambda x: self.smiles_to_img_png(x[compound_id_column],x[smiles_column],self.path_imgs),axis=1)
            logger.debug('Dataset created')

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        comp_id = row[self.compid_column]
        img_path = row['img_molecule']
        img_arr = cv2.imread(img_path)
        if random.random()>=0.50:
            angle = random.randint(0,359)
            rows, cols, channel = img_arr.shape
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            img_arr = cv2.warpAffine(img_arr, rotation_matrix, (cols, rows), cv2.INTER_LINEAR,
                                             borderValue=(255, 255, 255))  # cv2.BORDER_CONSTANT, 255)
        img_arr = np.array(img_arr) / 255.0
        img_arr = img_arr.transpose((2, 0, 1))
        label = row[self.label_column]

        return img_arr, label, comp_id
    
    def smiles_to_img_png(self, comp_id:str, smiles:str, output_path:str, IMG_SIZE:int=200, atomLabelFontSize:int=55, dotsPerAngstrom:int=100, bondLineWidth:int=1.5)->str:
        '''
        Given an id and an output path the function will create a image with the 2D molecule. 

        Returns: the path to the file i.e. /output_path/comp_id.png
        '''
        mol = Chem.MolFromSmiles(smiles)
        opt = MolDrawOptions()
        opt.atomLabelFontSize = atomLabelFontSize
        opt.dotsPerAngstrom = dotsPerAngstrom
        opt.bondLineWidth = bondLineWidth
        output_file = os.path.join(output_path, f"{comp_id}.png")
        Draw.MolToFile(mol, output_file, size= (IMG_SIZE, IMG_SIZE), options=opt)
        return output_file

class DEEPScreenDatasetPredict(Dataset):
    def __init__(self, path_tmp_files:str, df_compid_smiles_bioactivity:pd.DataFrame, smiles_column = 'smiles', compound_id_column = 'comp_id', bioactivity_label_column = 'bioactivity_label'):
            self.path = path_tmp_files
            self.path_imgs = os.path.join(self.path,'imgs')
            self.df = df_compid_smiles_bioactivity.copy()
            self.compid_column = compound_id_column
            self.smiles_column = smiles_column
            self.label_column  = bioactivity_label_column
            if not os.path.exists(self.path_imgs):
                os.makedirs(self.path_imgs)

            # creating molecules images -> path will be stored in 'img_molecule' column
            self.df['img_molecule'] = self.df.apply(lambda x: self.smiles_to_img_png(x[compound_id_column],x[smiles_column],self.path_imgs),axis=1)
            logger.debug('Dataset created')

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        comp_id = row[self.compid_column]
        img_path = row['img_molecule']
        img_arr = cv2.imread(img_path)
        img_arr = np.array(img_arr) / 255.0
        # img_arr = img_arr.transpose((2, 0, 1))
        return img_arr, comp_id


    def smiles_to_img_png(self, comp_id:str, smiles:str, output_path:str, IMG_SIZE:int=200, atomLabelFontSize:int=55, dotsPerAngstrom:int=100, bondLineWidth:int=1.5)->str:
        '''
        Given an id and an output path the function will create a image with the 2D molecule. 

        Returns: the path to the file i.e. /output_path/comp_id.png
        '''
        mol = Chem.MolFromSmiles(smiles)
        opt = MolDrawOptions()
        opt.atomLabelFontSize = atomLabelFontSize
        opt.dotsPerAngstrom = dotsPerAngstrom
        opt.bondLineWidth = bondLineWidth
        output_file = os.path.join(output_path, f"{comp_id}.png")
        Draw.MolToFile(mol, output_file, size= (IMG_SIZE, IMG_SIZE), options=opt)
        return output_file

def save_best_model_predictions(output_trained_model_path, experiment_name, epoch, validation_scores_dict, test_scores_dict, model, target_id, str_arguments,
                                                                                   all_test_comp_ids, test_labels, test_predictions):

    if not os.path.exists(os.path.join(output_trained_model_path, experiment_name)):
        os.makedirs(os.path.join(output_trained_model_path, experiment_name))

    output_file = "{}/{}/{}_best_val-{}-state_dict.pth".format(output_trained_model_path, experiment_name,
                                                                               target_id, str_arguments)
    torch.save(model.state_dict(),output_file)
               
    # print(all_test_comp_ids)
    str_test_predictions = "CompoundID\tLabel\tPred\n"
    for ind in range(len(all_test_comp_ids)):
        str_test_predictions += "{}\t{}\t{}\n".format(all_test_comp_ids[ind],
                                                          test_labels[ind],
                                                          test_predictions[ind])
    best_test_performance_dict = test_scores_dict
    best_test_predictions = str_test_predictions
    return validation_scores_dict, best_test_performance_dict, best_test_predictions, str_test_predictions, output_file

def dataloaders_generator(path_tmp_files:str, df_compid_smiles_bioactivity:pd.DataFrame, loader_type:str, bioactivity_label_column, train_batch_size=32, )->torch.utils.data.DataLoader:
    if loader_type == 'predict':
        logger.debug('dataloaders generator predict launched')
        dataset = DEEPScreenDatasetPredict(path_tmp_files, df_compid_smiles_bioactivity)
        return torch.utils.data.DataLoader(dataset)
    
    elif loader_type == 'train_random_split':
        logger.debug('dataloaders generator train_random_split launched')
        train, validate, test = np.split(df_compid_smiles_bioactivity.sample(frac=1, random_state=RANDOM_STATE), [int(.6*len(df_compid_smiles_bioactivity)), int(.8*len(df_compid_smiles_bioactivity))])
        logger.debug('datasets splited')
        logger.debug(f'Dataloaders created sizes (t/v/t) {len(train.index)}/{len(validate.index)}/{len(test.index)}')
        training_dataset = DEEPScreenDatasetTrain(path_tmp_files, train,compound_id_column='comp_id', bioactivity_label_column=bioactivity_label_column)
        validation_dataset = DEEPScreenDatasetTrain(path_tmp_files, validate, bioactivity_label_column=bioactivity_label_column)
        test_dataset = DEEPScreenDatasetTrain(path_tmp_files, test, bioactivity_label_column=bioactivity_label_column)

        train_sampler = SubsetRandomSampler(range(len(training_dataset)))
        train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=train_batch_size,
                                                  sampler=train_sampler)

        validation_sampler = SubsetRandomSampler(range(len(validation_dataset)))
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=train_batch_size,
                                                   sampler=validation_sampler)

        test_sampler = SubsetRandomSampler(range(len(test_dataset)))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=train_batch_size,
                                                   sampler=test_sampler)
        return train_loader, validation_loader, test_loader
    
    else:
        logger.error(f"Wrong data split mode")
        raise RuntimeError(f"'{loader_type}' is not a data split mode available, try 'train_random_split'")

def predict(target_id:str,trained_model_path:str, df_to_predict, fully_layer_1, fully_layer_2, drop_rate )->pd.Series:
    '''
    given a trained model in .pth from (path), and a path to a folder with molecules to predict, this function will return a dictionary with compounds ID as keys and 0/1 according to its prediction.
    '''

    device = get_device()

    data_loader = dataloaders_generator(df_to_predict,loader_type='predict')

    # loading model
    model = CNNModel1(fully_layer_1, fully_layer_2, drop_rate).to()
    model.load_state_dict(trained_model_path)
    model.eval()

    prediction = pd.Series(name=target_id)
    with torch.no_grad():
        for data in data_loader:
            img_arrs, comp_ids = data
            img_arrs = torch.tensor(img_arrs).type(torch.FloatTensor).to(device)
            y_pred = model(img_arrs).to(device)
            _, preds = torch.max(y_pred, 1)
            prediction[comp_ids[0]] = preds

    return prediction

def train(training_df:pd.DataFrame, target_id:str, result_files_path:str, tmp_files_path:str, fully_layer_1, fully_layer_2, learning_rate, batch_size, drop_rate, n_epoch, experiment_name, train_split_mode = 'train_random_split',  model_name = 'CNNModel1'):
    
    arguments = [str(argm) for argm in
                 [target_id, model_name, fully_layer_1, fully_layer_2, learning_rate, batch_size, drop_rate, n_epoch, experiment_name]]
    str_arguments = "-".join(arguments)
    logger.debug(f"Training:{str_arguments}")

    device = get_device()
    exp_path = os.path.join(result_files_path, "experiments", experiment_name)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    best_val_test_result_fl = open(
        "{}/best_val_test_performance_results-{}.txt".format(exp_path,str_arguments), "w")
    best_val_test_prediction_fl = open(
        "{}/best_val_test_predictions-{}.txt".format(exp_path,str_arguments), "w")

    train_loader, valid_loader, test_loader = dataloaders_generator(tmp_files_path, training_df, loader_type = train_split_mode, bioactivity_label_column=target_id,train_batch_size=batch_size)
    model = None
    if model_name == "CNNModel1":
        model = CNNModel1(fully_layer_1, fully_layer_2, drop_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    best_val_mcc_score, best_test_mcc_score = 0.0, 0.0
    best_val_test_performance_dict = dict()
    best_val_test_performance_dict["MCC"] = 0.0

    for epoch in range(n_epoch):
        total_training_count = 0
        total_training_loss = 0.0
        print("Epoch :{}".format(epoch))
        model.train()
        batch_number = 0
        all_training_labels = []
        all_training_preds = []
        print("Training mode:", model.training)
        for i, data in enumerate(train_loader):
            batch_number += 1
            # print(batch_number)
            # clear gradient DO NOT forget you fool!
            optimizer.zero_grad()
            img_arrs, labels, comp_ids = data
            img_arrs, labels = torch.tensor(img_arrs).type(torch.FloatTensor).to(device), torch.tensor(labels).to(device)

            total_training_count += len(comp_ids)
            y_pred = model(img_arrs).to(device)
            _, preds = torch.max(y_pred, 1)
            all_training_labels.extend(list(labels))
            all_training_preds.extend(list(preds))

            loss = criterion(y_pred.squeeze(), labels)
            total_training_loss += float(loss.item())
            loss.backward()
            optimizer.step()
        print("Epoch {} training loss:".format(epoch), total_training_loss)
        training_perf_dict = dict()
        try:
            training_perf_dict = prec_rec_f1_acc_mcc(all_training_labels, all_training_preds)
        except:
            print("There was a problem during training performance calculation!")
        # print(training_perf_dict)
        model.eval()
        with torch.no_grad():  # torch.set_grad_enabled(False):
            print("Validation mode:", not model.training)

            total_val_loss, total_val_count, all_val_comp_ids, all_val_labels, val_predictions = calculate_val_test_loss(model, criterion, valid_loader, device)
            
            val_perf_dict = dict()
            val_perf_dict["MCC"] = 0.0
            try:
                val_perf_dict = prec_rec_f1_acc_mcc(all_val_labels, val_predictions)
            except:
                print("There was a problem during validation performance calculation!")
            

            total_test_loss, total_test_count, all_test_comp_ids, all_test_labels, test_predictions = calculate_val_test_loss(
                model, criterion, test_loader, device)
            
            test_perf_dict = dict()
            test_perf_dict["MCC"] = 0.0
            try:
                test_perf_dict = prec_rec_f1_acc_mcc(all_test_labels, test_predictions)
            except:
                print("There was a problem during test performance calculation!")

            if val_perf_dict["MCC"] > best_val_mcc_score:
                best_val_mcc_score = val_perf_dict["MCC"]
                best_test_mcc_score = test_perf_dict["MCC"]

                validation_scores_dict, best_test_performance_dict, best_test_predictions, str_test_predictions, output_pth_file = save_best_model_predictions(exp_path,
                    experiment_name, epoch, val_perf_dict, test_perf_dict,
                    model, target_id, str_arguments,
                    all_test_comp_ids, all_test_labels, test_predictions)

        if epoch == n_epoch - 1:
            score_list = get_list_of_scores()
            for scr in score_list:
                best_val_test_result_fl.write("Test {}:\t{}\n".format(scr, best_test_performance_dict[scr]))
            best_val_test_prediction_fl.write(best_test_predictions)

            best_val_test_result_fl.close()
            best_val_test_prediction_fl.close()
        
    return output_pth_file


