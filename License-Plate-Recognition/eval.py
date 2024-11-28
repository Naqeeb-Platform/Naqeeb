precisionT = test_results.results_dict['metrics/precision(B)']
recallT = test_results.results_dict['metrics/recall(B)']
mAP50T = test_results.results_dict['metrics/mAP50(B)']
f1_scoreT = 2 * (precisionT * recallT) / (precisionT + recallT)

labelsT = ['Precision', 'Recall', 'mAP@50', 'F1 Score']
valuesT = [precisionT, recallT, mAP50T, f1_scoreT]
