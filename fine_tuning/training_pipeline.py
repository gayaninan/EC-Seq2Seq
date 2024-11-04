from fine_tuning.dependencies import install_dependencies
from fine_tuning.data_utils import load_and_split_dataset
from fine_tuning.model_utils import initialize_model, get_training_arguments, train_model, save_model_to_hub

def run_training_pipeline(model_type='bart', model_size='base', objective='summarisation'):
    '''Executes the full training pipeline with Hugging Face integration.'''
    install_dependencies()
    train_dataset, val_dataset = load_and_split_dataset(objective=objective)
    model = initialize_model(model_type=model_type, model_size=model_size)
    
    output_dir = f"./results/{model_type}_{model_size}_{objective}"
    training_args = get_training_arguments(output_dir=output_dir)
    
    trainer = train_model(model, train_dataset, val_dataset, training_args)
    save_model_to_hub(model, training_args.output_dir)
