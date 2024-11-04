import yaml
import pandas as pd
from evaluation.wer_evaluator import evaluate_model
from evaluation.results_generator import generate_results
from fine_tuning.config import CONFIG

scenarios = CONFIG['evaluation_scenarios']

test_data = [
    ("Have you noticed any changes in your weight?", "Have you noticed any changes in your wit?"),
    ("And I know that youve been on fortnightly Adalimumab", "Andi I know that youve been on fortnightly Adelaida map")
]

evaluation_results = []


for scenario in scenarios:
    model_type = scenario['model_type']
    model_size = scenario['model_size']
    objective = scenario['objective']

    model_name = CONFIG['models'][model_type][model_size]


    results_df = generate_results(test_data, model_name, model_type=model_type)


    wer_score = evaluate_model(model_name, test_data, model_type=model_type)

    evaluation_results.append({
        'Model Type': model_type,
        'Model Size': model_size,
        'Objective': objective,
        'WER Score': wer_score
    })

    results_df.to_csv(f'./results/{model_type}_{model_size}_{objective}_predictions.csv', index=False)
    print(f"Saved predictions for {model_type} {model_size} with objective {objective}")

summary_df = pd.DataFrame(evaluation_results)
summary_df.to_csv('./results/evaluation_summary.csv', index=False)
print("Evaluation summary saved to ./results/evaluation_summary.csv")
