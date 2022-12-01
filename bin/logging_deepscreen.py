import logging
import datetime


logging.basicConfig(filename = f'/home/sebastian-wsl/DEEPScreen/bin/log/{datetime.datetime.now()}.log',
                    level = logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s: %(message)s')
 

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S"))
handler.setLevel(logging.DEBUG)

logger = logging.getLogger('general')
logger.addHandler(handler)
