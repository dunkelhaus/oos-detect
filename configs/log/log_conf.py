LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'brief': {
            'class': 'logging.Formatter',
            'style': '{',
            'datefmt': '%I:%M:%S',
            'format': '{levelname:8s}; {name:<15s}; {message:s}'
        },
        'single-line': {
            'class': 'logging.Formatter',
            'style': '{',
            'datefmt': '%I:%M:%S',
            'format': '{levelname:8s}; {asctime:s}; {name:<15s} {lineno:4d}; {message:s}'
        },
        'multi-process': {
            'class': 'logging.Formatter',
            'style': '{',
            'datefmt': '%I:%M:%S',
            'format': '{levelname:8s}; {process:5d}; {asctime:s}; {name:<15s} {lineno:4d}; {message:s}'
        },
        'multi-thread': {
            'class': 'logging.Formatter',
            'style': '{',
            'datefmt': '%I:%M:%S',
            'format': '{levelname:8s}; {threadName:5d}; {asctime:s}; {name:<15s} {lineno:4d}; {message:s}'
        },
        'verbose': {
            'class': 'logging.Formatter',
            'style': '{',
            'datefmt': '%I:%M:%S',
            'format': '{levelname:8s}; {process:5d}; {threadName:8s}; {asctime:s}; {name:<15s} {lineno:4d}; {message:s}'
        },
        'multiline': {
            'class': 'logging.Formatter',
            'style': '{',
            'datefmt': '%I:%M:%S',
            'format': '{levelname:8s}\n{process:5d}\n{threadName:8s}\n{asctime:s}\n{name:<15s}{lineno:4d}\n{message:s}\n'
        }
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'brief',
            'stream' : 'ext://sys.stdout'
        },
        'info_file_handler': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'single-line',
            'filename': 'C:\Users\skjena\code\oos-detect\logs\info.log',
            'maxBytes': 1048576,
            'backupCount': 5,
            'mode': 'a',
            'encoding': 'utf-8'
        },
        'debug_file_handler': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'verbose',
            'filename': 'C:\Users\skjena\code\oos-detect\logs\debug.log',
            'maxBytes': 1048576,
            'backupCount': 5,
            'mode': 'a',
            'encoding': 'utf-8'
        },
    },
    'loggers': {
        'root': {
            'handlers': [
                'console',
                'debug_file_handler',
                'info_file_handler'
            ],
            'level': 'DEBUG',
            'propagate': False
        },
        '__main__': {
            'handlers': [
                'console',
                'debug_file_handler',
                'info_file_handler'
            ],
            'level': 'DEBUG',
            'propagate': False
        },
    }
}
