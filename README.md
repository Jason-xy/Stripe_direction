# Stripe_direction
条纹运动方向识别

## 纹背景下方向识别

##### 1. 任务

设计一个条纹背景下方向识别程序，能够实时显示摄像头相对条纹移动方向。

##### 2. 要求

（1） 

|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | ![clip_image001.png](https://wpcos-1300629776.cos.ap-chengdu.myqcloud.com/Gallery/2021/02/28/clip_image001.png) |


需要打印条纹纸，黑白条纹疏密不限，宽度不限，间隔相等。



（2）地上铺上条纹纸，使用手机拍摄，屏幕上不能有其他背景，只能出现  条纹，要求屏幕里至少有20根黑色条纹（所以距离越近，需要打印的纸张  越少，条纹需要越密），然后手机相对条纹移动，移动时间不小于10s。

（3）编写电脑程序，不限制使用语言，要求把手机移动方向标记出来。

##### 4. 备注

1. 至少做出一维移动下的方向识别，10s内至少折返5次。

2. 然后尝试二维移动及旋转的方式识别。

3. 开学我们将提供另外一段视频给你们测试，该测试视频，分辨率未知，帧数未知，背景干扰未知，要求算法一定要有很强的适应性。