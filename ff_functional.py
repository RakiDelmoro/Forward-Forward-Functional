import torch
from layer import ff_layer
from utils import manipulate_pixel_base_on_label

def forward_forward_network(feature_layers, activation_function, lr, threshold, epochs, device):
    layers = []

    for feature in range(len(feature_layers)-1):
        input_feature = feature_layers[feature]
        output_feature = feature_layers[feature+1]
        layer = ff_layer(in_features=input_feature, out_features=output_feature, activation_function=activation_function, learning_rate=lr, threshold=threshold, epochs=epochs, device=device)
        layers.append(layer)

    def training(positive_data, negative_data):
        positive_phase, negative_phase = positive_data, negative_data
        for forward_pass_training, _ in layers:
            positive_phase, negative_phase = forward_pass_training(positive_phase, negative_phase)

    def predicting(x: torch.Tensor) -> torch.Tensor:
        goodness_per_label = []
        for label in range(10):
            input_for_layer = manipulate_pixel_base_on_label(x, label, 10)
            layers_goodness = []
            for _, forward_pass_predicting in layers:
                layer_output = forward_pass_predicting(input_for_layer)
                calculate_layer_goodness = layer_output.pow(2).mean(1)
                layers_goodness.append(calculate_layer_goodness)
                input_for_layer = layer_output
            goodness_per_label.append(sum(layers_goodness).unsqueeze(1))
        prediction_scores = torch.cat(goodness_per_label, dim=1)
        return prediction_scores

    return training, predicting
