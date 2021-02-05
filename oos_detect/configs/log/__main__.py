import logging.config
from configs.log.log_conf import LOGGING_CONFIG

# --- Universal logger setup - startup task ---
logging.config.dictConfig(LOGGING_CONFIG)

# Logger setup.
log = logging.getLogger(__name__)
log.debug("Logging is configured.")


def testing_logging():
    """
    Testing logging.
    """
    print("Trying logging.")
    log.debug("Here's a message. Did it work?")


testing_logging()
