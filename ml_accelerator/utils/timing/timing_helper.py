# -- coding: utf-8 --
from ml_accelerator.config.params import Params
from ml_accelerator.utils.logging.logger_helper import get_logger
import time
from functools import wraps
from typing import Callable


# Get logger
LOGGER = get_logger(
    name=__name__,
    level=Params.LEVEL,
    txt_fmt=Params.TXT_FMT,
    json_fmt=Params.JSON_FMT,
    filter_lvls=Params.FILTER_LVLS,
    log_file=Params.LOG_FILE,
    backup_count=Params.BACKUP_COUNT
)


def _func_full_name(func: Callable):
    if not func.__module__:
        return func.__name__
    return "{a}.{b}".format(a=func.__module__, b=func.__name__)


def _human_readable_time(elapsed: float):
    mins, secs = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)

    if hours > 0:
        return "{a} hour {b} min {c} sec".format(a=int(round(hours, 0)), b=mins, c=round(secs, 2))
    elif mins > 0:
        return "{a} min {b} sec".format(a=int(round(mins, 0)), b=round(secs, 2))
    elif secs >= 0.1:
        return "{a} sec".format(a=round(secs, 2))
    else:
        return "{a} ms".format(a=int(round(secs * 1000.0, 0)))


def timing(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t1
        
        LOGGER.info(
            "Run time: %s ran in %s",
            _func_full_name(func), 
            _human_readable_time(elapsed)
        )
        # print("Run time: {a} ran in {b}".format(a=_func_full_name(func), b=b))
        return result

    return wrapper