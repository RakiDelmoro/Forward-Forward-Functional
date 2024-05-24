import gzip
import torch
import pickle
from utils import manipulate_pixel_base_on_label, get_wrong_label
from ff_functional import forward_forward_network
from features import tensor
from functools import partial
from torch.utils.data import TensorDataset, DataLoader

def main():
    EPOCHS = 10000000
    BATCH_SIZE = 2048
    LEARNING_RATE = 0.0001
    WIDTH = 28
    HEIGHT = 28

    # load training data into memory
    with (gzip.open('./training-data/mnist.pkl.gz', 'rb')) as file:
        ((training_input, training_expected), (validation_input, validation_expected), _) = pickle.load(file, encoding='latin-1')
	# convert numpy arrays to tensors
    training_input, training_expected, validation_input, validation_expected = map(tensor, (training_input, training_expected, validation_input, validation_expected))
    input_feature_size = HEIGHT * WIDTH

    image, label = next(iter(DataLoader(TensorDataset(training_input, training_expected), batch_size=BATCH_SIZE, shuffle=True)))
    validation_dataloader = DataLoader(TensorDataset(validation_input, validation_expected), batch_size=1, shuffle=True)
    
    label_for_negative_data = get_wrong_label(batched_label=label, num_classes=10)
    positive_data = manipulate_pixel_base_on_label(batched_image=image, batched_label=label, num_classes=10)
    negative_data = manipulate_pixel_base_on_label(batched_image=image, batched_label=label_for_negative_data, num_classes=10)

    model_training, model_predicting = forward_forward_network(feature_layers=[input_feature_size, 2000, 2000], activation_function=torch.nn.functional.relu, lr=0.01, threshold=2.0, epochs=10, device="cuda")
    model_training(positive_data, negative_data)
    print(model_predicting(image))

main()