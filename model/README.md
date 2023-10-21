[【计算机视觉】如何用传统方法在小样本数据集上实现语义分割模型](https://pengyizhang.github.io/2020/04/14/traditionalsegmentation/)

- 选择合适的像素特征描述子
- 选择合适的分类器
- 用提取的特征以及提供的标签对分类器进行训练

[【计算机视觉】评估语义分割精确度的指标](https://pengyizhang.github.io/2020/04/09/segmetric/)

- 像素精度 (Pixel Accuracy, PA)

  指的是预测正确的像素数量占所有像素的比例

  $$
  PA = \frac{\sum_{i=0}^kp_{ii}}{\sum_{i=0}^k\sum_{j=0}^k{p_{ij}}}
  $$
- 均像素精度 (Mean Pixel Accuracy, MPA)

  计算每个类内被正确分类的像素比例，然后取平均

  $$
  MPA = \frac{1}{k+1}\sum_{i=0}^k\frac{p_{ii}}{\sum_{j=0}^kp_{ij}}
  $$
- 均交并比 (Mean Intersection over union, MIoU)

  计算两个集合的交集和并集之比，语义分割中`真实值`与`预测值`两类

  $$
  MIoU = \frac{1}{k+1}\sum_{i=0}^k\frac{p_{ii}}{\sum_{j=0}^kp_{ij}+\sum_{j=0}^kp_{ji}-p_{ii}}
  $$
- Dice Coefficient (F1 Score)

  2倍的集合交集与两个集合总和的比值

  $$
  F1\_Score = \frac{\sum_{i=0}^kp_{ii}}{\sum_{i=0}^k\sum_{j=0}^k(p_{ij} + p{_{ji}})}
  $$

[【评价指标的代码实现】参考 `paddleseg`, `AdaptSegNet/compute_iou.py` 等](https://github.com/wasidennis/AdaptSegNet/blob/master/compute_iou.py)
