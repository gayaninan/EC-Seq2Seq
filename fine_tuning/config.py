import yaml

def load_config():
    '''Loads configurations from config.yml.'''
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    return config

# Load the full configuration
CONFIG = load_config()

# Access configuration components
HUGGINGFACE_MODEL_REPO = CONFIG['huggingface']['model_repo']
HUGGINGFACE_API_TOKEN = CONFIG['huggingface']['api_token']
MODEL_NAMES = CONFIG['models']
OBJECTIVES = CONFIG['objectives']
TRAINING_ARGS = CONFIG['training']
