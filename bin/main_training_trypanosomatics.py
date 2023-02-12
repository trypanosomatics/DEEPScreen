from deepscreen_trypanosomatics_update import trainer
import os
import pandas as pd

directory = 'DEEPScreen/training_files'
training_files_directory = os.listdir(directory)
training_files_directory_csv = list()
for file in training_files_directory:
    if file.find('.csv') == -1:
        continue
    df = pd.read_csv(os.path.join(directory,file), index_col = 0, header = 0)
    trainer_deepscreen = trainer(df,'deepscreen.db')
    trainer_deepscreen.train('./results', tmp_imgs=True)