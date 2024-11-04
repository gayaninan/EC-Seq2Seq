from datasets import load_metric

def calculate_wer(references, predictions):
    '''Calculates the Word Error Rate (WER) between references and predictions.'''
    wer_metric = load_metric('wer')
    wer_score = wer_metric.compute(predictions=predictions, references=references)
    return wer_score

def evaluate_model(model_name, test_data, model_type='bart'):
    '''Evaluates the model using WER on the provided test data.'''
    from error_correction import error_correct

    references = [ref for ref, _ in test_data]
    transcriptions = [trans for _, trans in test_data]
    predictions = [error_correct(trans, model_name, model_type=model_type)[0] for trans in transcriptions]
    wer_score = calculate_wer(references, predictions)

    return wer_score
