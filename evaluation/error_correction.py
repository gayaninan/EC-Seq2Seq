from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BartForConditionalGeneration, BartTokenizer

def error_correct(text, model_name, model_type='bart'):
    '''Performs error correction on text using a specified model.'''
    if model_type == 'bart':
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)
    elif model_type == 't5':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    input_ids = tokenizer.encode(text, return_tensors='pt', add_special_tokens=True)
    generated_ids = model.generate(input_ids=input_ids, 
                                   num_return_sequences=5, 
                                   num_beams=5, 
                                   max_length=512, 
                                   no_repeat_ngram_size=2, 
                                   repetition_penalty=3.5, 
                                   length_penalty=1.0, 
                                   early_stopping=True)

    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    return preds
