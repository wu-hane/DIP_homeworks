# 本次作业补全了泊松图像编辑和Pix2Pix算法。
## Pix2Pix算法补全了若干个卷积层和反卷积层，限于算力，只训练了100轮，误差表如下：
![](Pix2Pix/the_lose.png 'the_lose')
## 训练效果见"train_result文件"

## 对于泊松图像编辑，初步实现了部分功能，尚未实现混合梯度，下面是三张图片的运行结果：
![](data_poission\equation\image.png 'eg1')

![](data_poission\water\image2.png 'eg2')

![](data_poission\monolisa\image3.png 'eg3')

## 在模拟过程中，发现当图片若“过分”相似，则会返回模糊图片
![](data_poission\equation\image1.png 'eg4')