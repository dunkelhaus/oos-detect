import logging.config
from configs.log import LOGGING_CONFIG

# Universal logger setup - startup task.
logging.config.dictConfig(LOGGING_CONFIG)
