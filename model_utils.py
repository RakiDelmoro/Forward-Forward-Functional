import numpy as np
import statistics
from features import GREEN, RESET, RED

def print_correct_prediction(correct_prediction_list, number_to_print):
    print(f"{GREEN}Correct prediction!{RESET}")
    for i in range(number_to_print):
        each_item = correct_prediction_list[i]
        predicted, expected = each_item['predicted'], each_item['expected']
        print(f"Predicted: {predicted} Expected: {expected}")

def print_wrong_prediction(wrong_prediction_list, number_to_print):
    print(f"{RED}Wrong prediction!{RESET}")
    for i in range(number_to_print):
        each_item = wrong_prediction_list[i]
        predicted, expected = each_item['predicted'], each_item['expected']
        print(f"Predicted: {predicted} Expected: {expected}")

def print_percentile_of_correct_probabilities(probabilities_list, epoch_idx):
    tenth_percentile = np.percentile(probabilities_list, 10)
    ninetieth_percentile = np.percentile(probabilities_list, 90)
    average = statistics.fmean(probabilities_list)

    print(f"EPOCH: {epoch_idx} Average: {average} Tenth percentile: {tenth_percentile} Ninetieth percentile: {ninetieth_percentile}")

def runner(training_forward_pass, validation_forward_pass, positive_data, negative_data, validate_loader, epochs):
    list_of_digit_probabilities = []
    list_of_correct_prediction = []
    list_of_wrong_prediction = []
    for i in range(1, epochs+1):
        # Training layers
        training_forward_pass(positive_data, negative_data)
        # Validate model
        for image, expected in validate_loader:
            probability, predicted = validation_forward_pass(image)
            list_of_digit_probabilities.append(probability)
            if predicted == expected.item():
                predicted_and_expected = {'predicted': predicted, 'expected': expected.item()}
                list_of_correct_prediction.append(predicted_and_expected)
            else:
                predicted_and_expected = {'predicted': predicted, 'expected': expected.item()}
                list_of_wrong_prediction.append(predicted_and_expected)
        print_correct_prediction(list_of_correct_prediction, 5)
        print_wrong_prediction(list_of_wrong_prediction, 5)
        print_percentile_of_correct_probabilities(list_of_digit_probabilities, i)