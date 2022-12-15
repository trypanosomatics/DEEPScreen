import logging
import datetime

import os


import os
path = "./log"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(path)

logging.basicConfig(filename = f'../log/{datetime.datetime.now()}.log',
                    level = logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s: %(message)s')
 

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S"))
handler.setLevel(logging.DEBUG)

logger = logging.getLogger('general')
logger.addHandler(handler)
