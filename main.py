import gzip
import torch
import pickle
from features import tensor
from data_utils import load_data_to_memory, get_training_data, get_validation_data
from ff_functional import forward_forward_network
from torch.utils.data import TensorDataset, DataLoader

def main():
    BATCH_SIZE = 2048
    LEARNING_RATE = 0.01
    WIDTH = 28
    HEIGHT = 28

    # load training data into memory
    image_for_train, expected_for_train, image_for_validate, expected_for_validate = load_data_to_memory('./training-data/mnist.pkl.gz')
    training_data = get_training_data(image_for_train, expected_for_train, 10)
    validation_data = get_validation_data(image_for_validate, expected_for_validate)

    input_feature_size = HEIGHT * WIDTH
    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=1, shuffle=True)

    hidden_layers = [100] * 99
    feature_sizes = [input_feature_size] + hidden_layers
    model_runner = forward_forward_network(feature_layers=[784, 2000, 2000], activation_function=torch.nn.functional.relu, lr=LEARNING_RATE, threshold=2.0, device="cuda")
    model_runner(train_loader, validation_loader)

main()
