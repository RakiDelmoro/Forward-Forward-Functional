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
        for i, (training_layer, _) in enumerate(layers):
            positive_phase, negative_phase = training_layer(positive_phase, negative_phase)
            print(f"Training Layer: {i}")
        print(f"Finish train each layer!")

    def predicting(x: torch.Tensor):
        goodness_per_label = []
        for label in range(10):
            input_for_layer = manipulate_pixel_base_on_label(x, label, 10)
            goodness = []
            for _, layer in layers:
                output = layer(input_for_layer, False)
                goodness = goodness + [output.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        return goodness_per_label.argmax(dim=-1)
    
    return training, predicting

data_for_inference = torch.randn(1, 10, device="cuda")
positive_data = torch.randn(1, 10, device="cuda")
negative_data = torch.randn(1, 10, device="cuda")
model_training, model_predict = forward_forward_network([10, 10, 10, 10], torch.nn.functional.relu, 0.01, 2.0, 1000, "cuda")
model_training(positive_data, negative_data)
print(model_predict(data_for_inference))