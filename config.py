import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the chatbot application"""

    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    # Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH', 'chatbot_model.h5')
    WORDS_PATH = os.environ.get('WORDS_PATH', 'words.pkl')
    CLASSES_PATH = os.environ.get('CLASSES_PATH', 'classes.pkl')
    INTENTS_PATH = os.environ.get('INTENTS_PATH', 'intents.json')

    # External APIs
    OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY')

    # Training settings
    EPOCHS = int(os.environ.get('EPOCHS', 200))
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 5))
    LEARNING_RATE = float(os.environ.get('LEARNING_RATE', 0.01))

    # Response settings
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.25))
    MAX_RESPONSE_LENGTH = int(os.environ.get('MAX_RESPONSE_LENGTH', 500))

    # Conversation settings
    MAX_CONVERSATION_HISTORY = int(os.environ.get('MAX_CONVERSATION_HISTORY', 10))
    ENABLE_CONTEXT_AWARENESS = os.environ.get('ENABLE_CONTEXT_AWARENESS', 'True').lower() == 'true'

    # CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')

    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'chatbot.log')

    # Database settings (for future use)
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///chatbot.db')

    # API settings
    API_RATE_LIMIT = os.environ.get('API_RATE_LIMIT', '100 per hour')

    # Deployment settings
    PORT = int(os.environ.get('PORT', 5000))
    HOST = os.environ.get('HOST', '0.0.0.0')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    DATABASE_URL = 'sqlite:///test_chatbot.db'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
