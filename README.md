# tensorflow-笔记

1、tensorflow是一种符号语言，在执行run之前的所有网络搭建，都不会涉及到真正的运行
比如:

	a=tf.constant(10,dtype=tf.int32)
	b=tf.constant(10,dtype=tf.int32)
	c=a+b

这个时候c的值并不等于20。而是代表了一种操作，是让c=a+b，只有当需要运行的时候c才会有值，即运行：sess.run(c)
可以想象为在run之前的网络搭建是在做水管网络。管道搭好之后，要通水(run)才知道具体情况

2、tensorflow在网络搭建好后，在网络当中流动的东西是Tensor。	

	sess.run(X)

这句代码中。X必须是一个Tensor才行。否则会报错，比如这种：

	X=1
	with tf.Session() as sess:
		sess.run(X)

3、session是tensorflow当中的会话，一个网络必须要放在一个会话中才能运行。session可以在启动时设置某些配置，像这样：

	sess_config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    sess_config.gpu_options.allow_growth=True
	#设置sess是否打印每一层网络是放在GPU还是CPU中运行
	#以及当指定的GPU不存在时，可否用CPU帮忙运行
    with tf.Session(config=sess_config) as sess:

4、tensorflow当中的变量Variable要修改值，不能直接使用等号赋值。必须要使用assign来进行赋值。不然会导致Variable的性质改变成普通的Tensor，从而无法保存

5、tf.global_variables()可以查看当前所有变量，如果需要查看具体的值，可以使用
	
	sess.run(tf.global_variables())

6、保存某个网络，实际上是保存网络当中的Variables，可以简单的使用

	saver=tf.train.Saver()
	with tf.Session() as sess:
		saver.save(sess,"model/model.ckpt")

来保存网络中的所有变量，下次读取的时候使用：
	
	saver=tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess,"model/model.ckpt")

恢复所有变量

7、sess.run()可以除了必填一个Tensor的变量名，如果网络中有placeholder的占位符变量，还可以选填一个feed_dict作为占位符的输入。并且注意，并不是所有的占位符变量都需要feed，tensorflow会自动屏蔽掉没有填入的占位符相关的操作。这点在训练神经网络的时候很重要。有时候网络已经训练成功，只需要使用网络进行推理的时候，可以不在传入label，这个时候，跟label相关的最小化loss的操作都会失效。也就不会在改变网络中已经计算完成的变量了

7、以一个持续计算斐波那契数列的代码来演示，变量的修改，保存，读取，feed_dict。斐波那契的计算用的矩阵乘法：

$
\begin{bmatrix}
1&1\\ 
1&0
\end{bmatrix}^{n}\begin{bmatrix}
fib_{2}\\ 
fib_{1}
\end{bmatrix}=\begin{bmatrix}
fib_{n+2}\\ 
fib_{n+1}
\end{bmatrix}
$

代码如下：
首先设计斐波那契的类：

	# -*- coding: UTF-8 -*-
	import tensorflow as tf
	import numpy as np

	class Fib():
	    def __init__(self):
	        self.e=tf.constant([[1,1],[1,0]],dtype=tf.int32)
	        self.b=tf.constant(1,dtype=tf.int32)
	        self.init_mat=tf.placeholder(shape=[2,1],dtype=tf.int32)
	        self.init_step=tf.placeholder(shape=[],dtype=tf.int32)
	        self.result=tf.Variable([[1],[1]],dtype=tf.int32,name="result")
	        self.step=tf.Variable(0,dtype=tf.int32,name="step")
	        self.result_op=self.result.assign(tf.matmul(self.e,self.init_mat))
	        self.step_op=self.step.assign(self.init_step+self.b)
	        self.saver=tf.train.Saver()
	    def get_result(self,sess,init_mat):
	        return sess.run((self.result_op),feed_dict={self.init_mat:init_mat})
	    def get_step(self,sess,init_mat,init_step):
	        return sess.run((self.result_op,self.step_op),feed_dict={self.init_mat:init_mat,self.init_step:init_step})
	    def save_model(self,sess):
	        self.saver.save(sess,"model/model.ckpt")
	    def load_model(self,sess):
	        self.saver.restore(sess,"model/model.ckpt")
	        print(sess.run(tf.global_variables()))

这里step的作用主要是验证第七条的。我让get_result执行5次，但是get_step只执行一次，观察step和result的变化情况

	fib=Fib()
	with tf.Session() as sess:
	    tf.global_variables_initializer().run()
	    init_mat,init_step=sess.run(tf.global_variables())
	    for i in range(5):
	        init_mat=fib.get_result(sess,init_mat)
	        print(init_mat)
	    print(fib.get_step(sess,init_mat,init_step))
	    fib.save_model(sess)
	    print(sess.run(tf.global_variables()))

这里是网络的计算和存储，会计算到第五步的斐波那契

	fib=Fib()
	with tf.Session() as sess:
	    fib.load_model(sess)
	    init_mat,init_step=sess.run(tf.global_variables())
	    for i in range(5):
	        init_mat=fib.get_result(sess,init_mat)
	        print(init_mat)
	    print(fib.get_step(sess,init_mat,init_step))
	    print(sess.run(tf.global_variables()))

这里读取到上一次计算的result变量，接着计算第五步到第十步的斐波那契
注意这里不再需要使用

	tf.global_variables_initializer().run()

来初始化变量了，因为已经从ckpt中读取了上一次计算完成后变量的值
第一次运算的结果:

	[[2]
	 [1]]
	[[3]
	 [2]]
	[[5]
	 [3]]
	[[8]
	 [5]]
	[[13]
	 [ 8]]
	(array([[21],
	       [13]]), 1)
	[array([[21],
	       [13]]), 1]

第二次运算的结果：

	[array([[21],
	       [13]]), 1]
	[[34]
	 [21]]
	[[55]
	 [34]]
	[[89]
	 [55]]
	[[144]
	 [ 89]]
	[[233]
	 [144]]
	(array([[377],
	       [233]]), 2)
	[array([[377],
	       [233]]), 2]

