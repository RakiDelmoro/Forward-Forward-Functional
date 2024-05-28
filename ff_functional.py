import time
import torch
from layer import ff_layer
from utils import manipulate_pixel_base_on_label, get_wrong_label
from model_utils import print_correct_prediction, print_wrong_prediction, print_percentile_of_correct_probabilities

def forward_forward_network(feature_layers, activation_function, lr, threshold, epochs, device):
    layers = []
    layers_parameters = []

    for feature in range(len(feature_layers)-1):
        input_feature = feature_layers[feature]
        output_feature = feature_layers[feature+1]
        layer, w, b = ff_layer(in_features=input_feature, out_features=output_feature, activation_function=activation_function, device=device)
        layers.append(layer)
        layers_parameters.extend([[w,b]])

    def training_layer(positive_data, negative_data):
        print("Training...")
        for i, layer in enumerate(layers):
            optimizer = torch.optim.Adam(layers_parameters[i], lr)
            for _ in range(10):
                positive_output_features = layer(positive_data)
                negative_output_features = layer(negative_data)
                squared_average_activation_positive_phase = positive_output_features.pow(2).mean(dim=1)
                squared_average_activation_negative_phase = negative_output_features.pow(2).mean(dim=1)
                layer_loss = torch.log(1 + torch.exp(torch.cat([
                    -squared_average_activation_positive_phase+threshold,
                    squared_average_activation_negative_phase-threshold
                ]))).mean()
                print(f'\r{layer_loss}', end='', flush=True)
                time.sleep(1.0)
                optimizer.zero_grad()
                layer_loss.backward()
                optimizer.step()
            print()
            print(f"Layer {i+1} done training!")
            # Detach gradient flow to next layer
            positive_data, negative_data = layer(positive_data).detach(), layer(negative_data).detach()

    def predicting(x: torch.Tensor) -> torch.Tensor:
        print("Validating...")
        goodness_per_label = []
        for label in range(10):
            input_for_layer = manipulate_pixel_base_on_label(x, label, 10)
            layers_goodness = []
            for layer in layers:
                layer_output = layer(input_for_layer)
                calculate_layer_goodness = layer_output.pow(2).mean(1)
                layers_goodness.append(calculate_layer_goodness)
                input_for_layer = layer_output
            goodness_per_label.append(sum(layers_goodness).unsqueeze(1))
        prediction_scores = torch.cat(goodness_per_label, dim=1)
        return prediction_scores
    
    def runner(training_loader, validation_loader):
        for image, label in training_loader:
            positive_data = manipulate_pixel_base_on_label(image, label, 10)
            label_for_negative_data = get_wrong_label(label, 10)
            negative_data = manipulate_pixel_base_on_label(image, label_for_negative_data, 10)
            training_layer(positive_data, negative_data)

        list_of_prediction_probability = []
        list_of_correct_prediction = []
        list_of_wrong_prediction = []
        for test_image, test_label in validation_loader:
            prediction_score = predicting(test_image)
            model_prediction = prediction_score.argmax()
            probability = torch.nn.functional.softmax(prediction_score, dim=-1).max().item()

            list_of_prediction_probability.append(probability)
            if model_prediction.item() == test_label.item():
                predicted_and_expected = {'predicted': model_prediction.item(), 'expected': test_label.item()}
                list_of_correct_prediction.append(predicted_and_expected)
            else:
                predicted_and_expected = {'predicted': model_prediction.item(), 'expected': test_label.item()}
                list_of_wrong_prediction.append(predicted_and_expected)

        print_correct_prediction(list_of_correct_prediction, 5)
        print_wrong_prediction(list_of_wrong_prediction, 5)
        print_percentile_of_correct_probabilities(list_of_prediction_probability)

    return runner
