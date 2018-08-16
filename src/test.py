import os
import logging

home = os.path.dirname(os.getcwd())

log_file = os.path.join(home, "logs", f"log.log")
log_format = '[%(asctime)s]\t[%(levelname)s]\t[%(filename)s]\t[%(funcName)s]\t%(message)s'

logging.basicConfig(filename=log_file, level=logging.DEBUG, format=log_format)
status_logger = logging.getLogger("Status")

status_logger.info("Were")