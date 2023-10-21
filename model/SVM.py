from datasets import *
from pathlib import Path

from sklearn import svm, metrics
from PIL import Image
import numpy as np
from sklearn.externals import joblib


def data_train(model, img_train, mask_train, val_img, val_mask):
    iters = 0
    for img_path_train, mask_path_train in (img_train, mask_train):
        iters = iters + 1
        img = Image.open(img_path_train)
        mask = Image.open(mask_path_train)
        img_array = np.array(img).reshape([img.width * img.height, 3])
        mask_array = np.array(mask).reshape([mask.width * mask.height, 1])
        model.fit(img_array, mask_array)
        if iters % 5 == 0:
            joblib.dump(model, f'model_{iters}.pkl')
            data_val(model, val_img, val_mask)
    return model


def data_val(model, img_path, mask_path):
    for path_img, path_mask in (img_path, mask_path):
        img = Image.open(path_img)
        img_array = np.array(img).reshape([img.width * img.height, 3])
        mask = Image.open(path_mask)
        mask_array = np.array(mask).reshape([mask.width * mask.height, 1])
        predict = model.predict(img_array)
        # 转换成二维数组
        img_array = img_array.reshape([img.width, img.height, 3])
        mask_array = mask_array.reshape([mask.width, mask.height])
        predict = predict.reshape([mask.width, mask.height])
        loss_dice = metrics.f1_score(mask_array, predict)
        loss_iou = datasets.intersection_over_union(predict, mask_array)
        sum_loss_dice.append(loss_dice)
        sum_loss_iou.append(loss_iou)
    print(f"eval_iou_loss:", np.array(sum_loss_iou).sum() / len(sum_loss_iou))
    print(f"eval_dice_loss:", np.array(sum_loss_dice).sum() / len(sum_loss_dice))


def data_test(model, img_test, mask_test):
    sum_loss_dice = []
    sum_loss_iou = []
    for img_path_test, mask_path_test in (img_test, mask_test):
        img = Image.open(img_path_test)
        mask = Image.open(mask_path_test)
        img_array = np.array(img).reshape([img.width * img.height, 3])
        mask_array = np.array(mask).reshape([mask.width * mask.height, 1])
        predict = model.predict(img_array)
        # 转换成二维数组
        img_array = img_array.reshape([img.width, img.height, 3])
        mask_array = mask_array.reshape([mask.width, mask.height])
        predict = predict.reshape([mask.width, mask.height])
        loss_dice = metrics.f1_score(mask_array, predict)
        loss_iou = datasets.intersection_over_union(predict, mask_array)
        sum_loss_dice.append(loss_dice)
        sum_loss_iou.append(loss_iou)
    print("iou_loss:", np.array(sum_loss_iou).sum() / len(sum_loss_iou))
    print("dice_loss:", np.array(sum_loss_dice).sum() / len(sum_loss_dice))


def main():
    img_path = Path(__file__).parent / "dataset_ml"
    train_img, test_img, val_img, train_mask, test_mask, val_mask = datasets.load_data_ml(img_path)

    model = svm.SVC(kernel="poly", degree=3)
    model = data_train(model, train_img, train_mask, val_img, val_mask)
    data_test(model, test_img, test_mask)


if __name__ == "__main__":
    main()
