
# 本次作业基本实现了MLS算法初步。
  在my_mls.py中，实现了基础的mls点变换，在这个文档中，定义了一个Deformation类，类的输入参数依次是控制点、目标点、alpha值、绕动值。在这个类中定义了三个方法，这三个方法均以点为参数输入，获得对应变换的结果：
  
  1.aff_def:实现点的仿射变换

  2.sim_def:实现点的相似变换

  3.rig_def:实现点的刚性变换

本次作业demo实现了刚性变换，因为未优化，运行速度较慢，附件附有demo图片。
下面是效果图:

![](demo_mls.png 'demo_mls.png')


预计下一步是对代码进行优化