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
            positive_phase, negative_phase, avg_loss = forward_pass_training(positive_phase, negative_phase)
            print(f"Layer {i} average loss: {avg_loss}")
        # print(f"Finish train each layer!")

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
        character_probability = goodness_per_label.max(dim=-1)[0].item()
        indices = goodness_per_label.argmax(dim=-1).item()
        return character_probability, indices
    
    return training, predicting
