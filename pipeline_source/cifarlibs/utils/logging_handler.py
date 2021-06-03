import logging
from logging.config import dictConfig

logging_config = dict(
    version=1,
    formatters={
        'format': {'format': '%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s',
                   'datefmt': '%Y-%m-%d %H:%M:%S'
                   },
        'console': {'format': '%(asctime)s - %(funcName)s - %(levelname)s - %(message)s',
                   'datefmt': '%Y-%m-%d %H:%M:%S'
                   }
    },
    handlers={
        'handler': {'class': 'logging.StreamHandler',
                    'formatter': 'console',
                    'level': logging.INFO},
        'filehandler': {'class': "logging.FileHandler",
                        'formatter': 'format',
                        'level': logging.INFO,
                        'filename': "/tmp/cifar_log.txt"},
    },
    root={
        'handlers': ['handler', 'filehandler'],
        'level': logging.INFO,
    },
    disable_existing_loggers=False,
)


dictConfig(logging_config)
logger = logging.getLogger()
