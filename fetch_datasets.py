"""Fetches datasets required to train an image inpainting CNN.

Uses the high res DIV2K dataset (https://data.vision.ee.ethz.ch/cvl/DIV2K/).
"""

import pathlib
import os
import zipfile
import argparse

DATA_DIR = './data'

train_data_url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'
valid_data_url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'


def downloadAndUnzip(zip_url, zip_path, extract_dir):
    """Downloads the zip file at the given URL and extracts it.

    Args:
        zip_url: URL of zip file to download
        zip_path: local path to download zip file to
        extract_dir: local directory to extract zip file to
    """
    # Download
    print('Downloading {} to {}'.format(zip_url.split('/')[-1], zip_path)
    zip_path = tf.keras.utils.get_file(origin=zip_url,
                                       fname=zip_path)
    # Extract
    print('Extracting {} to {}'.format(zip_path, extract_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

def main(args):
    data_dir = pathlib.Path(args.data_dir).absolute()

    # Download and extract training images
    downloadAndUnzip(train_data_url,
                     str(data_dir/'train_images.zip'),
                     str(data_dir/'DIV2K_train_HR'))

    # Download and extract validation images
    downloadAndUnzip(valid_data_url,
                     str(data_dir/'valid_images.zip'),
                     str(data_dir/'DIV2K_valid_HR'))

    train_image_count = len(list((data_dir/'DIV2K_train_HR').glob('*.png')))
    print('Downloaded {} training images.'.format(train_image_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default=DATA_DIR,
                        help='Directory to store data.')

    args = parser.parse_args()

    main(args)
