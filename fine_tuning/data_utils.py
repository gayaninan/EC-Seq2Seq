from datasets import load_dataset
from fine_tuning.config import OBJECTIVES, TRAINING_ARGS

def load_and_split_dataset(objective='summarisation'):
    '''Loads and splits the dataset from Hugging Face based on the specified objective.'''
    dataset_name = OBJECTIVES.get(objective)
    if dataset_name is None:
        raise ValueError(f"No dataset found for objective '{objective}'. Please check config.yml for valid objectives.")
    
    dataset = load_dataset(dataset_name, split='train')
    dataset = dataset.train_test_split(test_size=TRAINING_ARGS['test_size'])
    return dataset['train'], dataset['test']
