from typing import Optional

from lwj_tools.utils.common import get_logger

logger = None


def log(message: str, rank: int = -1, log_file_path: Optional[str] = None):
    if rank != 0:
        return
    global logger
    if logger is None:
        logger = get_logger(name="Distill", log_path=log_file_path)
    logger.info(message)
