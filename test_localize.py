"""Evaluates the model on the test data"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
from utils.utils import Params, set_logger, load_checkpoint
import model.data_loader as data_loader
from utils.localization_utils import box_transform_inv
from utils.localization_utils import to_2d_tensor
from model.net import ModelWrapper, metrics, loss_fn

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/LocalizationDataset', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def test(model, dataloader, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        params: (Params) hyper parameters
    """

    # set model to evaluation mode
    model.eval()
    results = np.empty((0, 4), np.float)

    # compute metrics over the test dataset
    for i, (data_batch, labels_batch, original_shapes_batch) in enumerate(dataloader):
        # move to GPU if available
        if params.cuda:
            data_batch = data_batch.cuda(async=True)
            original_shapes_batch = original_shapes_batch.cuda(async=True)

        # fetch the next evaluation batch
        data_batch = Variable(data_batch)
        original_shapes_batch = Variable(original_shapes_batch)

        # compute model output
        output_batch = model(data_batch)

        result = box_transform_inv(output_batch, original_shapes_batch)

        result = result.data.cpu().numpy()
        results = np.append(results, result, axis=0)

    return results


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()  # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    set_logger(os.path.join(args.model_dir, 'test.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("getting the test dataloader - done.")

    # Define the model
    modelWrapperObj = ModelWrapper()
    model = modelWrapperObj.get_resnet18_network().cuda() if params.cuda else modelWrapperObj.get_resnet18_network()

    logging.info("Starting testing")

    # Reload weights from the saved file
    load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_results = test(model, test_dl, params)
    test_results_rounded = np.round(test_results, decimals=2)
    save_path = os.path.join(args.model_dir, "outputs.txt")
    np.savetxt(save_path, test_results_rounded, delimiter=',', fmt='%.3f')
