# 课题



## 思考方向

```
## 2019-07-15
	1. 考虑使用无监督学习进行聚类再进行有监督的学习；
	2. 时间间隔问题，由于数据预处理中间删除大量重复数据可能XXX 风险；
	3. 

```









## 进展

### 工作安排与进展

#### 第一周： 2019-07-24-数据处理
	1. 准备使用以下维度 ： [city、dev-type、net-type、name、time、alarm_level]
	2. city：东莞、中山、云浮、佛山、广州、惠州、梅州、深圳、清远、湛江、省NOC、茂名、阳江、韶关、揭阳、潮州、河源、肇庆、汕尾、汕头、江门、珠海 
	3. dev-type: AGCF、BAC、CCF、CSCF、HSS、IMGW、IMS、IOMC、MGCF、MGW、MMTEL、MRFP、SLF、SSF、CDMA、MSCE、OMC、CGOMU、ENS、OSG、SPG、SSFMGW、UPORTAL、云总机、HLRe
	4. net-type: CDMA、IMS (其他抛弃)
	5. name: 暂时不知道怎么处理，在考虑要不要这个维度
	6. time：处理成前后告警时间间隔(在网络中用 3 位表示)
	7. alarm_level: 1位表示
	
	问题： city 跟dev-type 维度太多，如果全部表示就需要53 位one-hot 。考虑怎么在网络中表示，初步想法是city 先做count 排序，从 1 开始到22 数量越大越后面。 



#### 2019-07-25-one-hot

	1. 由于ne_type字段只有两种，并且跟设备有关系，所以修改数据维度为：[city、dev-type、name、time、alarm_level]，删除net-type字段
	2. 继续使用CNN方式处理使用 one-hot 编码
	3. 处理数据，将所有数据归结到一张表里，分析数据
	4. 初步进行训练，使用 1M 数据进行测试，发现矩阵非常稀疏，计算过程较慢

![image-20190729115908845](http://ww4.sinaimg.cn/large/006tNc79gy1g5gkoufkamj30zk03zdip.jpg)

城市数据：

![](http://ww1.sinaimg.cn/large/006tNc79gy1g5gkavks3ij31ak0sg42o.jpg)

设备类型告警百分比：

![image-20190729115711875](http://ww2.sinaimg.cn/large/006tNc79gy1g5gkmtlmylj30v40u0qa7.jpg)



设备告警分布：

![image-20190729115734324](http://ww2.sinaimg.cn/large/006tNc79gy1g5gknc92ywj31iu0p4dmw.jpg)





#### 2019-07-26-entityEmbedding

	1. 修改数据组织方式，使用x_batch=100, y_batch=30进行训练
	2. 调研entityEmbedding方式对数据进行向量化
	3. 阅读entityEmbedding相关论文发现这种方式能有效处理结构化数据

![image-20190729120512453](../../Library/Application%20Support/typora-user-images/image-20190729120512453.png)

![image-20190729120608507](http://ww2.sinaimg.cn/large/006tNc79gy1g5gkw48cdcj31240jijuf.jpg)

#### 2019-07-27-中间数据结构

	1. 编写代码处理数据，将数据进行分类，对每一个设备进行编码；
	2. 尝试实现entityEmbedding
	3. 优化中间数据结构
![image-20190729121004810](http://ww4.sinaimg.cn/large/006tNc79gy1g5gl07nfa9j312k076t9z.jpg)

#### 2019-07-28-keras

	1. 优化中间数据结构，降低数据维度
	2. 换成keras，实现keras 网络
	3. 调试代码
![image-20190729114107175](http://ww3.sinaimg.cn/large/006tNc79gy1g5gk650vr4j30za06wjsn.jpg)





#### 第二周： 2019-07-29~2019-08-03

~~~
## 工作：
0. 使用one-hot编码直接进行训练，由于使用数据量、计算量过大，代码设计不合理，中间断电没有产出中间结果。
1. 使用 keras 构建embedding 层，减少数据的维度，加快训练进程，另外可以让数据更加稠密，训练效果会更加好
2. 重新处理数据为keras embedding层所需格式：参考图 1，各个字段使用不同的维度；
3. 使用keras 构建embedding 层时遇到问题，keras无法在embedding 时处理三维的数据，改用pytorch
4. 使用pytorch 完成数据lable、embedding、序列化等操作 图2 -4 
5. 构建新module、进行新模型训练；

## 存在问题
1. 使用MSELoss 的方式在这种数据格式下不合理
2. 网络结构太简单，学习效果不理想
3. accuracy评估方式无法区分设备名和时间两个维度
4. 目前数据embedding全部使用统一维度，设备724->64 ,设备类型 256->64... 浪费了空间
5. 
## 下周安排：
1. 调研loss 函数以及评估方法
2. 使用 keras 的思路，先分开embedding ，之后再拼接向量 （主要工作内容）
3. 开始进行所有数据测试
4. 优化网络结构（主要研究方向）
~~~

图 1： 使用keras 构建embedding 层：输入为张量维度，输出ebdding 结果的张量维度；无法处理理想的数据格式，也没有新的思路准备使用较熟悉的 pytorch

![image-20190805201009622](http://ww3.sinaimg.cn/large/006tNc79gy1g5p27w735ij31520u0tfj.jpg)



图 2：数据LabelEncoder，将设备名、设备类型等encode 

![image-20190805202450561](http://ww1.sinaimg.cn/large/006tNc79gy1g5p2n6feecj310m0n242b.jpg)





图 3： 重制数据格式

![image-20190805195223382](http://ww1.sinaimg.cn/large/006tNc79gy1g5p1ph59xgj312w0pi0wp.jpg)

图 4：将数据组装成模型数据

![image-20190805202617431](http://ww1.sinaimg.cn/large/006tNc79gy1g5p2pob1ysj31140p8jut.jpg)



图 5：开始训练，每500 个数据输出loss 情况，loss 开始收敛

![image-20190805204324887](http://ww1.sinaimg.cn/large/006tNc79gy1g5p36ilz8dj31n20r2dqg.jpg)



整体流程图：

![image-20190820161616642](../../Library/Application%20Support/typora-user-images/image-20190820161616642.png)



#### 第三周：2019-08-05

~~~
1. 发下代码问题，embedding 之后的数据，embedding 回去的值相差较大；
2. 修改代码，解决问题，进行CNN 模型验证
~~~

####  2019-08-19

~~~
1. 组会上昱航提出 x.sigmoid(x) 替代relu的方法，进行尝试
2. 取消1x1 的kernel 使用 3x3 加上padding=1 进行数据维度还原
3. 测试结果：在5w 数据集上比之前略有优化，速度也更快；
4. 重新审核代码
~~~

![image-20190820161858998](http://ww2.sinaimg.cn/large/006tNc79gy1g667xu9atej30v20ng0xr.jpg)

#### 2019-08-20

~~~~
1. 再次清洗数据，删除所有的‘告警恢复’，减少数据量，此类告警是无效告警；
2. 重新构建5w 数据进行预测，发现预测的告警种类会变多，从原来平均 32 到目前54 种；
3. 准确率没有明显下降，目前epoch=4 准确率在20%左右；
4. 猜想：目前应该从两个方面进行研究
		1. 从代码角度去看，是否数据处理仍有问题；
		2. 从CNN 模型角度出发，module 的设计是否有问题；
5. 是否花精力实现airbnb 论文 listing embedding ？
6. 考虑是否自己定制embedding；
7. 考虑先将一个x (128),y(64)的数据分别进行embedding ，之后再进行训练；（参考最新流程图）
~~~~

流程图更新后：

![image-20190820163034667](../../Library/Application%20Support/typora-user-images/image-20190820163034667.png)



#### 2019-08-21

~~~
1. 尝试使用seq2seq module ，开始编程实现；
2. 尝试其他loss function
3. 目前模型分析：
	1. 目前结果认为，算法仍然处在一个猜测的阶段，理论上结果准确率应该达到至少90%以上，目前仍停留在30%
	2. 认为在embedding 之后接入一层CNN 是一次失败的尝试；
4. 接下来的工作方向：
	1. 尝试使用其他的模型进行训练 ； 特别是GRU 和 seq2seq 模型
	2. 研究时间维度上是否均匀分布，如果是均匀分布则抛弃时间维度；
~~~

####2019-08-27

~~~
1. 暂停 min yang 方案；
2. seq2seq 在设备名维度上取得较好效果；模型保存为：pytorch_version-seq2seq-trained.chkpt
3. 发布版本；
4. 接下来工作，使用设备名进行时间预测；
~~~

