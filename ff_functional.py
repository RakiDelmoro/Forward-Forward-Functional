import torch
import statistics
from layer import ff_layer
from utils import manipulate_pixel_base_on_label
from model_utils import print_correct_prediction, print_wrong_prediction, print_percentile_of_correct_probabilities

def forward_forward_network(feature_layers, activation_function, lr, threshold, patient_amount, device):
    layers = []
    layers_parameters = []

    for feature in range(len(feature_layers)-1):
        input_feature = feature_layers[feature]
        output_feature = feature_layers[feature+1]
        layer, w, b = ff_layer(in_features=input_feature, out_features=output_feature, activation_function=activation_function, device=device)
        layers.append(layer)
        layers_parameters.extend([[w,b]])

    def train_each_batch(dataloader, layer_optimizer, layer_index, layer, epochs):
        losses_of_each_batch = []
        for positive_data, negative_data in dataloader:
            positive_output_features = layer(positive_data)
            negative_output_features = layer(negative_data)
            squared_average_activation_positive_phase = positive_output_features.pow(2).mean(dim=1)
            squared_average_activation_negative_phase = negative_output_features.pow(2).mean(dim=1)
            layer_loss = torch.log(1 + torch.exp(torch.cat([
                -squared_average_activation_positive_phase+threshold,
                squared_average_activation_negative_phase-threshold
            ]))).mean()
            layer_optimizer.zero_grad()
            layer_loss.backward()
            layer_optimizer.step()
            # print(f'\r Epoch: {epochs} Training layer: {layer_index+1} loss: {layer_loss}', end='', flush=True)
            losses_of_each_batch.append(layer_loss.item())
        average_loss_for_whole_batch = statistics.fmean(losses_of_each_batch)
        # print()
        # print(f'\r Epoch: {epochs} Layer: {layer_index+1} average loss for each batch: {round(average_loss_for_whole_batch, 5)}', end='', flush=True)
        return average_loss_for_whole_batch
    
    def run_once(dataloader, layer):
        previous_output = []
        for positive_data, negative_data in dataloader:
            positive_output_features = layer(positive_data).detach()
            negative_output_features = layer(negative_data).detach()
            previous_output.append((positive_output_features, negative_output_features))

        return previous_output

    def training_layer(data_loader):
        print("Training...")
        for i, layer in enumerate(layers):
            bad_epoch = 0
            current_epoch = 0
            optimizer = torch.optim.Adam(layers_parameters[i], lr)
            best_loss = None
            previous_loss = None
            while True:
                current_loss = train_each_batch(dataloader=data_loader, layer_optimizer=optimizer, layer_index=i, layer=layer, epochs=current_epoch)
                if best_loss is None:
                    best_loss = current_loss
                elif current_loss < best_loss:
                    best_loss = current_loss
                    if previous_loss is not None:
                        if abs((previous_loss / current_loss) - 1) < 0.001:
                            bad_epoch += 1
                        else:
                            bad_epoch = 0
                else:
                    bad_epoch += 1

                if bad_epoch > patient_amount:
                    print()
                    print(f"Done training layer: {i+1} takes {current_epoch} loss: {current_loss}")
                    data_loader = run_once(data_loader, layer)
                    break
                
                current_epoch +=1
                previous_loss = current_loss

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

    def validating(data_loader):
        print('validating...')
        predictions_probabilities = []
        correct_predictions = []
        wrong_predictions = []
        model_predictions = []
        for test_image, test_label in data_loader:
            prediction_score = predicting(test_image)
            model_prediction = prediction_score.argmax()
            probability = torch.nn.functional.softmax(prediction_score, dim=-1).max().item()
            
            correct_or_wrong = model_prediction.eq(test_label).int().item()
            model_predictions.append(correct_or_wrong)

            predictions_probabilities.append(probability)
            if model_prediction.item() == test_label.item():
                predicted_and_expected = {'predicted': model_prediction.item(), 'expected': test_label.item()}
                correct_predictions.append(predicted_and_expected)
            else:
                predicted_and_expected = {'predicted': model_prediction.item(), 'expected': test_label.item()}
                wrong_predictions.append(predicted_and_expected)

        print_correct_prediction(correct_predictions, 5)
        print_wrong_prediction(wrong_predictions, 5)
        print_percentile_of_correct_probabilities(predictions_probabilities)
        
        correct_prediction_count = model_predictions.count(1)
        wrong_prediction_count = model_predictions.count(0)
        correct_percentage = (correct_prediction_count / len(model_predictions)) * 100
        wrong_percentage = (wrong_prediction_count / len(model_predictions)) * 100
        print(f"Correct percentage: {round(correct_percentage, 1)} Wrong percentage: {round(wrong_percentage, 1)}")
    
    def runner(training_loader, validation_loader):
        training_layer(training_loader)
        validating(validation_loader)

    return runner
