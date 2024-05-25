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

    def predicting(x: torch.Tensor):
        goodness_per_label = []
        for label in range(10):
            input_for_layer = manipulate_pixel_base_on_label(x, label, 10)
            goodness = []
            for _, forward_pass_predicting in layers:
                input_for_layer = forward_pass_predicting(input_for_layer)
                goodness = goodness + [input_for_layer.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        in_training = x.shape[0] > 1
        if in_training:
            return goodness_per_label.argmax(dim=-1)
        else:
            indices = goodness_per_label.argmax(dim=-1).item()
            character_probability = goodness_per_label.max(dim=-1)[0].item()

            return goodness_per_label.argmax(dim=-1), indices, character_probability
    
    return training, predicting
