import argparse
import numpy as np
import os
import logging

logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        datefmt='%H:%M:%S')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, nargs="?", default="kronecker")
    args = vars(parser.parse_args())

    logging.info(f"Numpifying folder {args['folder']}")

    files_to_convert = []

    for filename in os.listdir(args['folder']):
        if filename.endswith('.el'):
            input_file = os.path.join(args['folder'], filename)
            output_file = os.path.join(args['folder'], f'{filename[:-3]}.npy')
            if not os.path.exists(output_file):
                files_to_convert.append((input_file, output_file))

    logging.info(f"Number of files to convert: {len(files_to_convert)}")
    for i, (in_path, out_path) in enumerate(files_to_convert):
        logging.info(f"[{i+1}/{len(files_to_convert)}] converting: '{in_path}'")
        data = np.genfromtxt(in_path, dtype=np.int32)
        np.save(out_path, data)
