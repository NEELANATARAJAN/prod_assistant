from .custom_logger import CustomLogger
# Create singleton instance of CustomLogger to be used across the application
GLOBAL_LOGGER = CustomLogger().get_logger("prod_assistant")
