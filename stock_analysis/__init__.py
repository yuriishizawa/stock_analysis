import sys

from loguru import logger

STDOUT_FORMAT = "[<g>{time:HH:mm:ss}</g> <lvl>{level}</lvl>] <b>{message}</b> <d><c>({file}:{line})</c></d>"  # noqa: E501
logger.add(sys.stdout, format=STDOUT_FORMAT, level="DEBUG")
