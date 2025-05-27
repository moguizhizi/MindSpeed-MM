import logging


def get_logger(name):
    _logger = logging.getLogger(name)
    # 设置日志level
    _logger.setLevel("INFO")
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(module)s - %(lineno)d - %(levelname)s - %(message)s')
    # 控制台打印
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)
    return _logger
