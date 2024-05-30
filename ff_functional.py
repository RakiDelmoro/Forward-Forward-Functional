import time
import torch
import statistics
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

    def training_layer(data_loader):
        print("Training...")
        for i, layer in enumerate(layers):
            optimizer = torch.optim.Adam(layers_parameters[i], lr)
            best_loss = None
            bad_epoch = 0
            # Training each layer
            epochs = 0
            while True:
                batched_loss_for_given_layer = []
                # Train all batch
                for b, (image, label) in enumerate(data_loader):
                    positive_data = manipulate_pixel_base_on_label(image, label, 10)
                    label_for_negative_data = get_wrong_label(label, 10)
                    negative_data = manipulate_pixel_base_on_label(image, label_for_negative_data, 10)
                    first_layer = i == 0
                    if first_layer:
                        positive_output_features = layer(positive_data)
                        negative_output_features = layer(negative_data)
                    else:
                        previous_layer_positive_features = layers[i-1](positive_data).detach()
                        previous_layer_negative_features = layers[i-1](negative_data).detach()
                        positive_output_features = layer(previous_layer_positive_features)
                        negative_output_features = layer(previous_layer_negative_features)
                    squared_average_activation_positive_phase = positive_output_features.pow(2).mean(dim=1)
                    squared_average_activation_negative_phase = negative_output_features.pow(2).mean(dim=1)
                    layer_loss = torch.log(1 + torch.exp(torch.cat([
                        -squared_average_activation_positive_phase+threshold,
                        squared_average_activation_negative_phase-threshold
                    ]))).mean()

                    optimizer.zero_grad()
                    layer_loss.backward()
                    optimizer.step()
                    print(f'\r Epoch: {epochs} item in a batch: {b} Training layer: {i+1} loss: {layer_loss} {bad_epoch}', end='', flush=True)
                    time.sleep(1.0)
                    batched_loss_for_given_layer.append(layer_loss.item())
                average_loss = statistics.fmean(batched_loss_for_given_layer)
                epochs += 1
                if best_loss is None:
                    best_loss = average_loss
                elif average_loss < best_loss:
                    best_loss = average_loss
                    bad_epoch = 0
                else:
                    bad_epoch += 1
                # patient amount
                if bad_epoch > 5: break

    def validating(data_loader):
        list_of_prediction_probability = []
        list_of_correct_prediction = []
        list_of_wrong_prediction = []
        for test_image, test_label in data_loader:
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

    def predicting(x: torch.Tensor) -> torch.Tensor:
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
        training_layer(training_loader)
        validating(validation_loader)

    return runner
