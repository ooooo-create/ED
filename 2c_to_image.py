import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# get all `.2C` files

file_need_to_convert = Path().cwd().glob('*.2C')
all_2c_files = [x for x in file_need_to_convert]


# convert '.2C' to image(.jpg)
def to_convert(file):
    image_path = (file.parent / 'image' / file.stem).with_suffix('.jpg')
    with open(file, 'rb') as image_file:
        image_bytes = image_file.read()
    byte_array = bytearray(image_bytes[len(image_bytes) - 2352 * 1728 * 3:])
    numpy_array = np.frombuffer(byte_array, dtype=np.uint8).reshape((1728, 2352, 3))
    numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(image_path), numpy_array)


# test function 'to_convert'
def test_to_convert():
    file_to_test = all_2c_files[1]
    to_convert(file_to_test)


def pipeline():
    # test_to_convert()
    for file in tqdm(all_2c_files):
        to_convert(file)


if __name__ == "__main__":
    pipeline()
