import logging
import datetime

import os
path = "./log"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(path)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:'+'%(levelname)s'.rjust(8) + ' - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

date_now = str(datetime.datetime.now())[:19].replace(' ','_')

file_handler = logging.FileHandler(f'./log/{date_now}.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)


if (logger.hasHandlers()):
    logger.handlers.clear()

logger.propagate = False
    
logger.addHandler(file_handler)
logger.addHandler(console_handler)