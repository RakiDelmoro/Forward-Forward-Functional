import torch
import statistics
import numpy as np
from features import GREEN, RESET, RED
from utils import manipulate_pixel_base_on_label, get_wrong_label

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

def train(train_forward_pass, predict_forward_pass, training_loader):
    print("Training...")
    train_loss = 0.0
    for train_image, train_expected in training_loader:
        label_for_negative_data = get_wrong_label(batched_label=train_expected, num_classes=10)
        positive_data = manipulate_pixel_base_on_label(batched_image=train_image, batched_label=train_expected, num_classes=10)
        negative_data = manipulate_pixel_base_on_label(batched_image=train_image, batched_label=label_for_negative_data, num_classes=10)
        
        train_forward_pass(positive_data, negative_data) # Training layers
        model_prediction = predict_forward_pass(train_image).argmax(dim=-1)
        loss = 1.0 - model_prediction.eq(train_expected).float().mean().item()
        train_loss += loss

    return train_loss / len(training_loader)

def validate(predict_forward_pass, validation_loader):
    print("Validating...")
    list_of_prediction_probability = []
    list_of_correct_prediction = []
    list_of_wrong_prediction = []
    validate_loss = 0.0
    for image, expected in validation_loader:
        model_output_score = predict_forward_pass(image)
        model_prediction = model_output_score.argmax(dim=-1)
        probability_of_model_prediction = torch.nn.functional.softmax(model_output_score, dim=-1).max().item()

        list_of_prediction_probability.append(probability_of_model_prediction)
        if model_prediction.item() == expected.item():
            predicted_and_expected = {'predicted': model_prediction.item(), 'expected': expected.item()}
            list_of_correct_prediction.append(predicted_and_expected)
        else:
            predicted_and_expected = {'predicted': model_prediction.item(), 'expected': expected.item()}
            list_of_wrong_prediction.append(predicted_and_expected)

        loss = 1.0 - model_prediction.eq(expected).float().mean().item()
        validate_loss += loss

    print_correct_prediction(list_of_correct_prediction, 5)
    print_wrong_prediction(list_of_wrong_prediction, 5)
    print_percentile_of_correct_probabilities(list_of_prediction_probability)
    
    return validate_loss / len(validation_loader)

def runner(training_forward_pass, validation_forward_pass, training_loader, validate_loader, epochs):
    for epoch in range(1, epochs+1):
        train_loss = train(training_forward_pass, validation_forward_pass, training_loader)
        validate_loss = validate(validation_forward_pass, validate_loader)
        print(f"EPOCH {epoch}: Training loss: {train_loss} Validation loss: {validate_loss}")
