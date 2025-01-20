# loginit.py
import logging
import re
import os
from datetime import datetime
PROJ_BASEPATH = ""

# ANSI escape sequences for terminal colors
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Additional standard colors
    ORANGE = "\033[38;5;208m"
    PINK = "\033[38;5;213m"
    PURPLE = "\033[38;5;129m"
    LIGHT_BLUE = "\033[38;5;45m"
    LIGHT_GREEN = "\033[38;5;119m"
    LIGHT_CYAN = "\033[38;5;123m"
    LIGHT_RED = "\033[38;5;203m"
    LIGHT_MAGENTA = "\033[38;5;207m"

    # 添加加粗颜色
    BOLD_RED = BOLD + RED
    BOLD_GREEN = BOLD + GREEN
    BOLD_YELLOW = BOLD + YELLOW
    BOLD_BLUE = BOLD + BLUE
    BOLD_MAGENTA = BOLD + MAGENTA
    BOLD_CYAN = BOLD + CYAN
    BOLD_WHITE = BOLD + WHITE

    BOLD_ORANGE = BOLD + ORANGE
    BOLD_PINK = BOLD + PINK
    BOLD_PURPLE = BOLD + PURPLE
    
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    BACKGROUND_BLACK = "\033[40m"
    BACKGROUND_RED = "\033[41m"
    BACKGROUND_GREEN = "\033[42m"
    BACKGROUND_YELLOW = "\033[43m"
    BACKGROUND_BLUE = "\033[44m"
    BACKGROUND_MAGENTA = "\033[45m"
    BACKGROUND_CYAN = "\033[46m"
    BACKGROUND_WHITE = "\033[47m"
    
    BACKGROUND_BRIGHT_BLACK = "\033[100m"
    BACKGROUND_BRIGHT_RED = "\033[101m"
    BACKGROUND_BRIGHT_GREEN = "\033[102m"
    BACKGROUND_BRIGHT_YELLOW = "\033[103m"
    BACKGROUND_BRIGHT_BLUE = "\033[104m"
    BACKGROUND_BRIGHT_MAGENTA = "\033[105m"
    BACKGROUND_BRIGHT_CYAN = "\033[106m"
    BACKGROUND_BRIGHT_WHITE = "\033[107m"

class ProjectFilter(logging.Filter):
    def __init__(self, project_dir):
        self.project_dir = project_dir

    def filter(self, record):
        # 检查日志记录的文件路径是否属于项目目录
        return record.pathname.startswith(self.project_dir)

# Custom formatter class
class ColorFormatter(logging.Formatter):
    COLOR_MAP = {
        logging.DEBUG: Colors.BLUE + Colors.BOLD,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.MAGENTA,
    }

    MAX_LENGTH = 512

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelno, Colors.WHITE)
        # Parse color directive from the message if present
        if '{type:' in record.msg:
            record.msg, msg_color = self.parse_color_directive(record.msg)
            color = msg_color or color
        
        message = super().format(record)  

        if self.MAX_LENGTH and len(record.message) > self.MAX_LENGTH:
            msg = record.msg
            args = record.args
            record.msg = record.message[:self.MAX_LENGTH] + '...'
            record.args = ()
            message = super().format(record)  
            record.msg = msg
            record.args = args
        
        return f'{color}{message}{Colors.RESET}'
    
    def parse_color_directive(self, message):
        match = re.search(r'{type:(.*?)}', message)
        if match:
            color_name = match.group(1).strip().upper()
            color = getattr(Colors, color_name, None)
            if color:
                # Remove the color directive from the message
                message = message[:match.start()] + message[match.end():]
                return message, color
        return message, None


def setup_logging(max_length=512):
    global logger_initialized
    if logger_initialized:
        return
    logger_initialized = True
    
    # Ensure the log directory exists
    log_dir = os.path.join(PROJ_BASEPATH, "configs/debug/logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create log file name based on current time
    log_filename = datetime.now().strftime('log_%Y-%m-%d_%H-%M-%S.log')
    log_filepath = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.DEBUG)
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter for console output
    console_formatter = ColorFormatter('[%(name)s - %(levelname)s] - %(message)s (%(pathname)s:%(lineno)d)')
    console_formatter.MAX_LENGTH = max_length

    # Create formatter for file output without max length
    file_formatter = logging.Formatter('[%(name)s - %(levelname)s] - %(message)s (%(pathname)s:%(lineno)d)')
    # 创建日志过滤器实例
    project_filter = ProjectFilter("/home/legion4080/AIPJ/MYXY")

    # Set formatters for handlers
    ch.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)
    # 应用过滤器到控制台和文件处理器
    ch.addFilter(project_filter)
    file_handler.addFilter(project_filter)

    # Add the handlers to the logger
    logger.addHandler(ch)
    logger.addHandler(file_handler)

# Initialize a flag to ensure setup_logging runs only once
logger_initialized = False


