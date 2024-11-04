
from evaluation.error_correction import error_correct
import pandas as pd

def generate_results(test_data, model_name, model_type='bart'):
    '''Generates predictions using the specified model and returns them as a DataFrame.'''
    references = [ref for ref, _ in test_data]
    transcriptions = [trans for _, trans in test_data]
    predictions = [error_correct(trans, model_name, model_type=model_type)[0] for trans in transcriptions]

    results_df = pd.DataFrame({
        'reference': references,
        'transcription': transcriptions,
        'prediction': predictions
    })

    return results_df
