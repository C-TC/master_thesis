# Extract arrays from .npz file into separate .npy files
# Usage: python3 extract_npz_files.py fname.npz
import numpy as np
import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Incorrect number of arguments\nUsage: python3 extract_npz_files.py fname.npz")
        exit(-1)

    fname = sys.argv[1]
    fname_root = fname.split(".")[0]
    dir_name = f"{fname_root}_extracted"
    with np.load(fname) as data:
        print(f"Extracting into director: {dir_name}")
        if  not os.path.exists(dir_name):
            os.mkdir(dir_name)
        for f in data.files:
            print(f"Extracting {f}")
            np.save(f"{dir_name}/{f}.npy", data[f])
    

if __name__ == "__main__":
    main()



