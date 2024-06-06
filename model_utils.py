import torch
import statistics
import numpy as np
from features import GREEN, RESET, RED

def print_correct_prediction(correct_prediction_list, amount_to_print):
    print(f"{GREEN}Correct prediction!{RESET}")
    for i in range(amount_to_print):
        each_item = correct_prediction_list[i]
        predicted, expected = each_item['predicted'], each_item['expected']
        print(f"Predicted: {predicted} Expected: {expected}")

def print_wrong_prediction(wrong_prediction_list, amount_to_print):
    print(f"{RED}Wrong prediction!{RESET}")
    for i in range(amount_to_print):
        each_item = wrong_prediction_list[i]
        predicted, expected = each_item['predicted'], each_item['expected']
        print(f"Predicted: {predicted} Expected: {expected}")

def print_percentile_of_correct_probabilities(probabilities_list):
    tenth_percentile = np.percentile(probabilities_list, 10)
    ninetieth_percentile = np.percentile(probabilities_list, 90)
    average = statistics.fmean(probabilities_list)

    print(f"Average: {average} Tenth percentile: {tenth_percentile} Ninetieth percentile: {ninetieth_percentile}")
