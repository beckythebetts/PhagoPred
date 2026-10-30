# Submodules aren't imported here so that light scripts (e.g. dataset_creation)
# don't pull in tools' heavy dependencies; use `from PhagoPred.utils import tools`.
from PhagoPred.utils.logger import get_logger, get_log_path

__all__ = ['get_logger', 'get_log_path']
