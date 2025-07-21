import logging

# 1. Configure the logger (do this once at the start of your app)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 2. Get a logger instance in any module
logger = logging.getLogger(__name__)

# 3. Use the logger to record messages
logger.info("This is an informational message.")
logger.warning("This is a warning message.")
logger.error("This is an error message.")

# This debug message will not appear because the level is set to INFO
logger.debug("This is a debug message.")