import torch
import statistics
from layer import ff_layer
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
            losses_of_each_batch.append(layer_loss.item())
        average_loss_for_whole_batch = statistics.fmean(losses_of_each_batch)
        print(f'\r Epoch: {epochs} Layer: {layer_index+1} average loss for each batch: {round(average_loss_for_whole_batch, 5)}', end='', flush=True)
        return average_loss_for_whole_batch
    
    def forward_pass_once(dataloader, layer):
        previous_output = []
        for positive_data, negative_data in dataloader:
            positive_output_features = layer(positive_data).detach()
            negative_output_features = layer(negative_data).detach()
            previous_output.append((positive_output_features, negative_output_features))

        return previous_output

    def training_layer(data_loader):
        print("Training...")
        for layer_index, layer in enumerate(layers):
            bad_epoch = 0
            current_epoch = 0
            optimizer = torch.optim.Adam(layers_parameters[layer_index], lr)
            best_loss = None
            previous_loss = None
            while True:
                current_loss = train_each_batch(dataloader=data_loader, layer_optimizer=optimizer, layer_index=layer_index, layer=layer, epochs=current_epoch)
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
                    print(f"Done training layer: {layer_index+1}")
                    data_loader = forward_pass_once(data_loader, layer)
                    break
                previous_loss = current_loss
                current_epoch +=1
    
    def predicting(batched_images_with_combine_labels):
        batched_goodness_per_label = []
        for combined_label_in_image in batched_images_with_combine_labels:
            input_for_layer = combined_label_in_image
            layers_goodness = []
            for layer in layers:
                layer_output = layer(input_for_layer)
                layer_goodness = layer_output.pow(2).mean(1)
                input_for_layer = layer_output
                layers_goodness.append(layer_goodness)
            batched_layer_goodness = sum(layers_goodness)
            each_item_in_batch_per_label_goodness = batched_layer_goodness.view(batched_layer_goodness.shape[0], 1)
            batched_goodness_per_label.append(each_item_in_batch_per_label_goodness)
        batched_prediction_scores = torch.cat(batched_goodness_per_label, dim=1)
        return batched_prediction_scores

    def validating(batched_image, batched_label):
        print('validating...')
        batched_model_prediction = predicting(batched_image)

        predictions_probabilities = []
        correct_predictions = []
        wrong_predictions = []
        model_predictions = []
        for each_item in range(batched_model_prediction.shape[0]):
            model_prediction = batched_model_prediction[each_item]
            expected = batched_label[each_item]
            digit_predicted = model_prediction.argmax()
            digit_probability = torch.nn.functional.softmax(model_prediction, dim=0).max().item()
            
            correct_or_wrong = digit_predicted.eq(expected).int().item()
            model_predictions.append(correct_or_wrong)

            predictions_probabilities.append(digit_probability)
            if digit_predicted.item() == expected.item():
                predicted_and_expected = {'predicted': digit_predicted.item(), 'expected': expected.item()}
                correct_predictions.append(predicted_and_expected)
            else:
                predicted_and_expected = {'predicted': digit_predicted.item(), 'expected': expected.item()}
                wrong_predictions.append(predicted_and_expected)

        print_correct_prediction(correct_predictions, 5)
        print_wrong_prediction(wrong_predictions, 5)
        print_percentile_of_correct_probabilities(predictions_probabilities)
        
        correct_prediction_count = model_predictions.count(1)
        wrong_prediction_count = model_predictions.count(0)
        correct_percentage = (correct_prediction_count / len(model_predictions)) * 100
        wrong_percentage = (wrong_prediction_count / len(model_predictions)) * 100
        print(f"Correct percentage: {round(correct_percentage, 1)} Wrong percentage: {round(wrong_percentage, 1)}")
    
    def runner(training_loader, test_image, test_label):
        training_layer(training_loader)
        validating(test_image, test_label)

    return runner
