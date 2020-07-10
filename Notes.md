# darknet

## TODO
1. CSPNet
2. SPPNet

## 测试阶段相关问题

### 检测框的获取流程
1. 网络进行前向传播，每一层输出的结果都保存在`layer.output`属性中
2. 遍历所有`yolo`层，获取符合条件的box。<br>
    具体做法为：依次遍历每个cell、每个cell上的设置的每个anchor，保留`confidence>thresh`的box，其余过滤掉。对于保留下来的box，将每个类别的prob进行如下更新操作：`prob = prob*confidence > thresh ? prob*confidence : 0`
3. 对过滤后的box进行`nms`操作，nms逐类别进行，具体流程如下：
   1. 逐类别遍历
   2. 对当前类别按所有box的类别置信度从大到小排序
   3. 选出置信度最高的box
   4. 将剩余box与选出的box逐一进行IoU计算，若IoU>nms_thresh，则过滤掉。过滤方法为将box当前类别的prob置为0
4. 最终检测结果输出
    1. 依次遍历每个box
    2. 依次遍历box的每个类别，若类别的prob大于0，则输出这个box
    3. 一个box可能被输出多次，因为多个类别的prob有可能都满足条件

## yolov4相关信息

1. backbone: CSPNet的设计思想 + Residual block，v4-tiny中对[route]层进行了增强，对特征图进行分组输出
2. spp:
