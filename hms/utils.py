import logging
from logging.handlers import RotatingFileHandler
import os
FORMAT_STR = '%(asctime)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def configure_logging(dir_path, format_strs=[None], name='log', log_suffix=''):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    logger = logging.getLogger()  # root logger
    formatter = logging.Formatter(FORMAT_STR, DATE_FORMAT)
    file_handler = logging.FileHandler(filename="{0}/{1}.log".format(dir_path, name), mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if os.isatty(2):
        import coloredlogs
        coloredlogs.install(fmt=FORMAT_STR, level='INFO')
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
