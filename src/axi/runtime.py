"""CLI 与 Python PTC 共用的进程内工具状态。"""

from axi.config import app_config
from axi.executor import Executor
from axi.registry import Registry

registry = Registry(max_fields=app_config.search.max_fields)
executor = Executor(registry)
