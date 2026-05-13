import logging
import sys

def get_logger(name: str = "datathon") -> logging.Logger:
    """
    Returns a standard stdout logger for the Datathon pipeline.
    Ensures that logs are properly formatted and displayed in Colab.
    """
    logger = logging.getLogger(name)
    
    # Only configure if it doesn't already have handlers to avoid duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        
        logger.addHandler(ch)
        
    return logger
