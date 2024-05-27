import gzip
import torch
import pickle
from utils import manipulate_pixel_base_on_label, get_wrong_label
from ff_functional import forward_forward_network
from features import tensor
from functools import partial
from torch.utils.data import TensorDataset, DataLoader
from model_utils import runner

def main():
    EPOCHS = 1
    LAYER_EPOCHS = 100
    BATCH_SIZE = 2048
    LEARNING_RATE = 0.01
    WIDTH = 28
    HEIGHT = 28

    # load training data into memory
    with (gzip.open('./training-data/mnist.pkl.gz', 'rb')) as file:
        ((training_input, training_expected), (validation_input, validation_expected), _) = pickle.load(file, encoding='latin-1')
	# convert numpy arrays to tensors
    training_input, training_expected, validation_input, validation_expected = map(tensor, (training_input, training_expected, validation_input, validation_expected))
    
    input_feature_size = HEIGHT * WIDTH
    train_loader = DataLoader(TensorDataset(training_input, training_expected), batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(TensorDataset(validation_input, validation_expected), batch_size=1, shuffle=True)

    model_training, model_predicting = forward_forward_network(feature_layers=[input_feature_size, 2000, 2000], activation_function=torch.nn.functional.relu, lr=LEARNING_RATE, threshold=2.0, epochs=LAYER_EPOCHS, device="cuda")
    runner(model_training, model_predicting, train_loader, validation_loader, EPOCHS)

main()
