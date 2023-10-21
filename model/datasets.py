from pathlib import Path
from PIL import Image
import numpy as np
import cv2


# image/*.jpg
# mask/*.png

def load_data_ml(data_path: Path):
    img_list = [img for img in (data_path / "image").glob(".jpg")]
    mask_list = [mask for mask in (data_path / "mask").glob("*.png")]
    rate1, rate2 = 0.7, 0.2
    offset1 = int(len(data_list) * rate1)
    offset2 = int(len(data_list) * (rate1 + rate2))
    train_img = img_list[:offset1]
    test_img = img_list[offset1:offset2]
    val_img = img_list[offset2:]
    train_mask = mask_list[:offset1]
    test_mask = mask_list[offset1:offset2]
    val_mask = mask_list[offset2:]
    return train_img, test_img, val_img, train_mask, test_mask, val_mask


def data_transforms():
    pass


def intersection_over_union(pred, label, num_classes):
    assert pred.shape == label.shape
    iou = np.zeros(num_classes)
    for i in range(num_classes):
        intersection = ((pred == i) & (label == i)).sum()
        union = ((pred == i) | (label == i)).sum()
        if union > 0:
            iou[i] = intersection / union
        else:
            iou[i] = np.nan
    return iou


def all_to_gary(input_dir: Path, output_dir: Path = None, num_classes=None):
    """

    :param input_dir:
    :param output_dir:
    :param num_classes: {():(),():(),():()}
    :return:
    """
    input_list = [img for img in input_dir.glob(".png")]
    for input_img in input_list:
        to_gray(input_img,output_dir,num_classes)


def to_gray(input_img: Path, output_img=None, num_classes=None):
    img_name = input_img.name
    img = cv2.imread(str(input_img))
    for key, item in num_classes.items():
        img[np.all(img == item, axis=-1)] = key
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if output_img is None:
        output_path = output_img / img_name
    else:
        if not (input_img.parent / "new").exists():
            Path(input_img.parent / "new").mkdir()
        output_path = input_img.parent / "new" / img_name
    cv2.imwrite(str(output_path), img)


# to_gray(Path(
#     __file__).parent / "dataset_ml" / "image" / "CE3_BMYK_PCAML-C-002_SCI_N_20140113190142_20140113190142_0008_A.png",
#         {(0, 0, 0): (0, 0, 0), (1, 1, 1): (0, 0, 128)})