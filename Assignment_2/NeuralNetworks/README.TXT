说明：


1、此代码为matlab NN代码的python版本，由于作者能力有限，发现错误望及时指出


2、主测试文件为test.py，参数修改也在这个文件中


matlab的NN的readme：


---------------------------------------------------------------------------------------

test_fot_NN:主文件，包括神经网络的创建、训练、测试


1.nn_create:创建一个神经网络的结构体


2.nn_train:训练神经网络，包括


（1）nn_forward:前馈


（2）nn_backpropagation:反向传播


（3）牛顿梯度下降


3.nn_test:测试一个神经网络，包括


（1）nn_predict:由输入得到输出


----------------------------------------------------------------------------------------


3、可以用ipython notebook（推荐）复制测试文件中的代码到cell中去，运行nn_testChess()即可，如要使用请访问www.anaconda.com下载anaconda


4、也可在python环境中输入
    

from test import nn_testChess
    

nn_testChess()


5、或者注释掉test.py里的函数定义，命令行切换目录执行python test.py即可


6、运行代码需要的python包：numpy、matplotlib、sklearn