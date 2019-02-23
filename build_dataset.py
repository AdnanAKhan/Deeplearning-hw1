import argparse
import os
import pandas as pd
import numpy as np

SIZE = 224  # Resnet16
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.2
SEED = 230

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    default='raw_dataset/Localization dataset',
                    help="Directory with the Images and training and test csvs")
parser.add_argument('--output_dir', default='data/LocalizationDataset', help="Where to write the new data")


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the train and test dataset  directories
    train_images_txt = os.path.join(args.data_dir, 'train_images.txt')
    train_boxes_txt = os.path.join(args.data_dir, 'train_boxes.txt')
    test_images_txt = os.path.join(args.data_dir, 'test_images.txt')

    # Get the filenames in each directory (train and test)
    # filtering the images based on the file type. In this case, this code only except files with .jpg extension.
    number_training_images = 0
    with open(train_images_txt, 'r') as f:
        number_training_images = len(f.readlines())

    number_training_boxes = []
    with open(train_boxes_txt, 'r') as f:
        number_training_boxes = len(f.readlines())

    assert number_training_boxes == number_training_images, \
        'Number of training {}  and label (Boxes) {} are not same'.format(number_training_images,
                                                                          number_training_boxes)
    # load everything in pandas data frame for manipulation
    df_boxes = pd.read_csv(train_boxes_txt, sep=' ', header=None)
    df_names = pd.read_csv(train_images_txt, header=None)
    train_df = pd.concat([df_names, df_boxes], axis=1, ignore_index=True)
    train_df.columns = ['filename', 'x', 'y', 'w', 'h']

    test_df = pd.read_csv(test_images_txt, header=None)
    test_df.columns = ['filename']

    train_df.sort_values(by=['filename'], inplace=True)
    train_df.reindex(np.random.RandomState(seed=SEED).permutation(train_df.index))

    # Split the images in 'train' into 80% train and 20% val
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    split = int(TRAIN_SPLIT * len(train_df))

    train_filenames_with_label_df = train_df[:split]
    val_filenames_with_label_df = train_df[split:]

    dataset_df = {'train': train_filenames_with_label_df,
                  'val': val_filenames_with_label_df,
                  'test': test_df}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, val and test
    for split in ['train', 'val', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed csv file to {}".format(split, output_dir_split))

        df = dataset_df[split]
        df.to_csv(path_or_buf=os.path.join(output_dir_split, '{}.csv'.format('dataset')), index=False)

    print("Done building dataset")
