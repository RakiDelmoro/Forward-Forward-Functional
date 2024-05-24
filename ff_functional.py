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
        for i, (forward_pass_training, _) in enumerate(layers):
            output = forward_pass_training(positive_phase, negative_phase)
            positive_phase, negative_phase = output[0], output[1]
            print(f"Layer index {i} result")
        print(f"Finish train each layer!")

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
        return goodness_per_label.argmax(dim=-1)
    
    return training, predicting
