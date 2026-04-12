# Questify Configuration File

# Server Configuration
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000

# CORS Configuration
CORS_ORIGINS = ['http://localhost:8000', 'http://127.0.0.1:8000']

# Model Configuration
MODELS = {
    'summarization': {
        'model_name': 'facebook/bart-large-cnn',
        'device': -1,  # -1 for CPU, 0 for GPU (requires CUDA)
        'max_length': 150,
        'min_length': 50,
    },
    'mcq_generation': {
        'keywords_extract': 10,
        'num_distractors': 3,
        'max_questions': 20,
    }
}

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Input Validation
MIN_TEXT_LENGTH = 20  # minimum words
MAX_TEXT_LENGTH = 10000  # maximum words
MIN_QUESTIONS = 1
MAX_QUESTIONS = 20

# Timeout Configuration
REQUEST_TIMEOUT = 300  # seconds
