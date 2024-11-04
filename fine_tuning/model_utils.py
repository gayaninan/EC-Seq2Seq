from transformers import Trainer, TrainingArguments, BartForConditionalGeneration, T5ForConditionalGeneration
from fine_tuning.config import MODEL_NAMES, TRAINING_ARGS, HUGGINGFACE_MODEL_REPO, HUGGINGFACE_API_TOKEN

def initialize_model(model_type, model_size):
    '''Initializes a specified model based on type and size.'''
    model_name = MODEL_NAMES.get(model_type, {}).get(model_size)
    if not model_name:
        raise ValueError(f"Invalid model type '{model_type}' or size '{model_size}'.")
    
    if model_type == 'bart':
        return BartForConditionalGeneration.from_pretrained(model_name)
    elif model_type == 't5':
        return T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_training_arguments(output_dir, **kwargs):
    '''Configures and returns training arguments.'''
    training_args = TRAINING_ARGS.copy()
    training_args.update(kwargs)
    training_args['output_dir'] = output_dir
    training_args['evaluation_strategy'] = 'epoch'
    training_args['per_device_train_batch_size'] = training_args.pop('batch_size')
    training_args['per_device_eval_batch_size'] = training_args['per_device_train_batch_size']
    return TrainingArguments(**training_args)

def train_model(model, train_dataset, val_dataset, training_args):
    '''Initializes the Trainer and starts the training process.'''
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()
    return trainer

def save_model_to_hub(model, output_dir):
    '''Saves and pushes the fine-tuned model to Hugging Face Model Hub.'''
    model.save_pretrained(output_dir)
    print(f"Model saved locally to {output_dir}")
    
    print("Uploading to Hugging Face Hub...")
    model.push_to_hub(HUGGINGFACE_MODEL_REPO, use_auth_token=HUGGINGFACE_API_TOKEN)
    print(f"Model uploaded to Hugging Face Hub at {HUGGINGFACE_MODEL_REPO}")
