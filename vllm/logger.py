from __future__ import annotations

import logging


class _OnceLogger(logging.LoggerAdapter):
    def __init__(self, logger: logging.Logger):
        super().__init__(logger, {})
        self._once_messages: set[tuple[int, str]] = set()

    def info_once(self, msg, *args, **kwargs):
        key = (logging.INFO, str(msg))
        if key not in self._once_messages:
            self._once_messages.add(key)
            self.info(msg, *args, **kwargs)

    def warning_once(self, msg, *args, **kwargs):
        key = (logging.WARNING, str(msg))
        if key not in self._once_messages:
            self._once_messages.add(key)
            self.warning(msg, *args, **kwargs)


def init_logger(name: str) -> _OnceLogger:
    return _OnceLogger(logging.getLogger(name))

