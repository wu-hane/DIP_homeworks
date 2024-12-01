# 本次实验和GAN有关

 ## 第一个实验，是对上次Pix2Pix的一个改良，实验结果如下

   由于条件有限,选取Epoch=20

 ![](loss.jpg 'loss')

   结果如下

  ![](result_5.png 'result')

 ## 第二个实验是结合DragGAN和面部特征提取

   对于这个实验，较为两个部分较为复杂，故而只实现了初级的结合。

   由于项目已经开源,从而只给出有修改的模块。主要修改了DragGAN中drag_widget.py,visualizer_drag.py.以及新添加的两个模块。主要思路是先选取图片提取面部特征点。在图片上做出想要的修改，由于选择的点未必是特征点，于是我们提供了选项‘Run Warping’(由于框架直接套的第一次作业所以名称未改)将变换对应的向量的起点平移至最近的关键点，然后导出就可以获得变换的初始条件。由于DragGAN会实时渲染而本次使用的面部识别库时间处理时间较长，故创建了info_p.py存储数值减小系统负荷。最后的visualizer_drag.py将使用info_p.py中的相关信息作为变换的依据，不过代价是visualizer_drag.py原本的GUI交互失效较多。

   下面给出使用方法：

      先获取DragGAN库，将drag_widget.py,visualizer_drag.py各替换原文件；将其余块和visualizer_drag.py放置在同一目录下。

      先运行get_trans_info.py选取变换，然后导出，最后运行：

      .scripts\gui.bat

      开始变换，在合理时刻截图即可获得图片。

   ![](inl.png 'run get_info')

   ![](in2.png 'run bat')

   ![](00002.png 'cut')