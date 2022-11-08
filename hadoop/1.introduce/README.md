# Hadoop

​	Hadoop是一个由**Apache基金会**所开发的**分布式系统基础架构**。

​	创始人Doug Cutting

Google在大数据方面的三篇论文：

​	GFS--->HDFS

​	Map-Reduce--->MR

​	BigTable--->HBase



​	Hadoop三大发行版本：Apache(底层版本)、Cloudera、Hortonworks。

​	并行计算：高效性、高容错率、高可靠性、高扩展性



# 面试重点

​	Hadoop 1.x、2.x、3.x区别

​	Hadoop 解决两件事：海量数据的存储、海量数据的计算



## 1. Hadoop 1.x组成



<img src="C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221102101814088.png" alt="image-20221102101814088" style="zoom:67%;" />



## 2. Hadoop 2.x组成

​	资源调度管理着CPU和内存。

<img src="C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221102102309170.png" alt="image-20221102102309170" style="zoom:67%;" />



## 3.HDFS架构概述

​	Hadoop Distributed File System，简称HDFS，是一个分布式文件系统。

​		<img src="C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221102112557548.png" alt="image-20221102112557548" style="zoom:67%;" />

1 ) NameNode (nn):存储文件的**元数据**，如**文件名**，**文件目录结构**，**文件属性**（生成时间、副本数、文件权限)，以及每个文件的**块列表**和**块所在的DataNode**等。

2）DataNode(dn)在本地文件系统**存储文件块数据**以及**块数据的校验和**。

3 ) Secondary NameNode(2nn):每隔一段时间对NameNode元数据备份。



## 4.YARN架构概述

​	Yet Another Resource Negotiator简称YARN，另一种资源协调者，是Hadoop 的资源管理器。

​	![image-20221102115857524](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221102115857524.png)



## 5.MapReduce架构概述

MapReduce将计算过程分为两个阶段：Map和Reduce

1) Map阶段并行处理输入数据
2) Reduce阶段对Map结果进行汇总

![image-20221102162943496](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221102162943496.png)



## 6.HDFS、YARN、MapReduce三者关系



![image-20221102163749057](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221102163749057.png)



## 7.大数据技术生态体系



![image-20221102165623456](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221102165623456.png)



## 8.推荐系统项目框架

![image-20221102170205002](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221102170205002.png)



## 9. 配置多个主机

进入 su root

![image-20221102203651941](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221102203651941.png)

vim /etc/sysconfig/network-scripts/ifcfg-ens33   

​	改成static、增加ip地址。这里要看的很仔细很仔细

![image-20221103092918528](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221103092918528.png)

​	上面图片里面写错是GATEWAY（因为这个写错，我弄了一天，哭泣

vim /etc/hostname

![image-20221102205448881](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221102205448881.png)

vim /etc/hosts

![image-20221103184209970](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221103184209970.png)

ll 查看隐藏目录

reboot

用root用户登录

ifconfig 查看ip地址是不是192.168.10.100

systemctl restart network 重启网卡就好



## 

## 10. Xshell

 新建连接

![image-20221103190428239](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221103190428239.png)



![image-20221103190437322](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221103190437322.png)



再试一下几个常用的命令：

	1. hostname
	1. ifconfig
	1. 切换到普通用户 su hadoop100



![image-20221103193725770](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221103193725770.png)





# Hadoop环境准备

1）准备操作

![image-20221103194512707](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221103194512707.png)

（2）安装epel-release

注：Extra Packages for Enterprise Linux是为“红帽系”的操作系统提供额外的软件包，适用于**RHEL、CentOS和Scientific Linux**。相当于是一个软件仓库，大多数rpm包在官方 repository 中是找不到的）

[root@hadoop100 ~]# yum install -y epel-release

<img src="C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221103195602126.png" alt="image-20221103195602126" style="zoom:67%;" />

​	杀死进程：kill -9 3030

**关闭防火墙，关闭防火墙开机自启**

​	[root@hadoop100 ~]# systemctl stop firewalld

​	[root@hadoop100 ~]# systemctl disable firewalld.service

注意：在企业开发时，通常单个服务器的防火墙时关闭的。公司整体对外会设置非常安全的防火墙

一般在企业里面服务器之间的防火墙不开，整个服务器集群和外部网络才加防火墙。

**配置atguigu用户具有root权限，方便后期加sudo执行root权限的命令**

​	[root@hadoop100 ~]# vim /etc/sudoers

​	修改/etc/sudoers文件，在%wheel这行下面添加一行，如下所示：

​	## Allow root to run any commands anywhere

​	root  ALL=(ALL)   ALL

​	## Allows people in group wheel to run all commands

​	%wheel ALL=(ALL)    ALL

添加新的一行

<img src="C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221103201426092.png" alt="image-20221103201426092" style="zoom:67%;" />

​	退出当前用户第二种方式

​	![image-20221103203704011](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221103203704011.png)

​	现在要删掉上面的 rh 文件

![image-20221103204804321](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221103204804321.png)

​	安装两个文件夹 **module、software**

​	![image-20221103205024602](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221103205024602.png)

​	上面的两个文件的拥有者属于root，将文件 module文件夹 的拥有者设为 hadoop100，群体的使用者 hadoop100:

​		sudo chown hadoop100:hadoop100 module/ software/

​	![image-20221103210149432](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221103210149432.png)



**卸载虚拟机自带的JDK**

因为桌面版会自动安装JDK，所以需要卸载。

[root@hadoop100 ~]# rpm -qa | grep -i java | xargs -n1 rpm -e --nodeps 

Ø rpm -qa：查询所安装的所有rpm软件包

Ø grep -i：忽略大小写

Ø xargs -n1：表示每次只传递一个参数

Ø rpm -e –nodeps：强制卸载软件

![image-20221103210504493](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221103210504493.png)

删除java包->查看是否删除完成

![image-20221103220155387](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221103220155387.png)

reboot

## 1.2 克隆虚拟机

​	虚拟机克隆三个->更改主机名和IP地址

## 2.3 在hadoop102安装JDK

​	在hadoop102上安装jdk，然后在hadoop103、hadoop104上克隆jdk

**用XShell传输工具将JDK导入到opt目录下面的software文件夹下面**

![image-20221104082257263](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221104082257263.png)

​	tar -zxvf jdk-8u212-linux-x64.tar.gz -C /opt/module/

在Linux系统下的opt目录中查看软件包是否导入成功

​	[atguigu@hadoop102 ~]$ ls /opt/software/

**解压JDK到/opt/module目录下**

[atguigu@hadoop102 software]$ tar -zxvf jdk-8u212-linux-x64.tar.gz -C /opt/module/

**配置JDK环境变量**

（1）新建/etc/profile.d/my_env.sh文件

[atguigu@hadoop102 ~]$ sudo vim /etc/profile.d/my_env.sh

添加如下内容

\#JAVA_HOME

export JAVA_HOME=/opt/module/jdk1.8.0_212

export PATH=$PATH:$JAVA_HOME/bin

（2）保存后退出

:wq

（3）source一下/etc/profile文件，让新的环境变量PATH生效

​	[atguigu@hadoop102 ~]$ source /etc/profile

​	(可以看一下vim /etc/profile，里面会循环执行.sh脚本文件)

测试JDK是否安装成功

[atguigu@hadoop102 ~]$ java -version

如果能看到以下结果，则代表Java安装成功。

java version "1.8.0_212"



## 2.4 在hadoop102安装Hadoop

**2****）进入到Hadoop****安装包路径下**

[atguigu@hadoop102 ~]$ cd /opt/software/

**3****）解压安装文件到/opt/module****下面**

[atguigu@hadoop102 software]$ tar -zxvf hadoop-3.1.3.tar.gz -C /opt/module/

**4****）查看是否解压成功**

[atguigu@hadoop102 software]$ ls /opt/module/

hadoop-3.1.3

**5****）将Hadoop****添加到环境变量**

​    （1）获取Hadoop安装路径

[atguigu@hadoop102 hadoop-3.1.3]$ pwd

/opt/module/hadoop-3.1.3

（2）打开/etc/profile.d/my_env.sh文件

[atguigu@hadoop102 hadoop-3.1.3]$ sudo vim /etc/profile.d/my_env.sh

Ø 在my_env.sh文件末尾添加如下内容：（shift+g）

\#HADOOP_HOME

export HADOOP_HOME=/opt/module/hadoop-3.1.3

export PATH=$PATH:$HADOOP_HOME/bin

export PATH=$PATH:$HADOOP_HOME/sbin

Ø 保存并退出： :wq

（3）让修改后的文件生效

[atguigu@hadoop102 hadoop-3.1.3]$ source /etc/profile

**6****）测试是否安装成功**

[atguigu@hadoop102 hadoop-3.1.3]$ hadoop version

Hadoop 3.1.3

**1****）查看Hadoop****目录结构**

<img src="C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221104085937894.png" alt="image-20221104085937894" style="zoom:67%;" />



<img src="C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221104090229521.png" alt="image-20221104090229521" style="zoom:67%;" />



<img src="C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221104090652898.png" alt="image-20221104090652898" style="zoom: 50%;" />



<img src="C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221104091548227.png" alt="image-20221104091548227" style="zoom:80%;" />

**2****）重要目录**

（1）bin目录：存放对Hadoop相关服务（hdfs，yarn，mapred）进行操作的脚本

（2）etc目录：Hadoop的配置文件目录，存放Hadoop的配置文件

（3）lib目录：存放Hadoop的本地库（对数据进行压缩解压缩功能）

（4）sbin目录：存放启动或停止Hadoop相关服务的脚本

（5）share目录：存放Hadoop的依赖jar包、文档、和官方案例

# 第3章 Hadoop运行模式



Hadoop运行模式包括：**本地模式**、**伪分布式模式**以及**完全分布式模式**。

- [Local (Standalone) Mode](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html#Standalone_Operation) **本地模式**：单机运行，只是用来演示一下官方案例。生产环境不用。 数据存储在linux本地

- [Pseudo-Distributed Mode](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html#Pseudo-Distributed_Operation) **伪分布式模式：**也是单机运行，但是具备Hadoop集群的所有功能，一台服务器模拟一个分布式的环境。个别缺钱的公司用来测试，生产环境不用。 数据存储在HDFS

- [Fully-Distributed Mode](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html#Fully-Distributed_Operation) **完全分布式模式：**多台服务器组成分布式环境。生产环境使用。数据存储在HDFS/多台服务器工作

![image-20221104095233137](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221104095233137.png)



## 3.1 本地运行模式（官方WordCount）

**1****）创建在hadoop-3.1.3****文件下面创建一个wcinput****文件夹**

[atguigu@hadoop102 hadoop-3.1.3]$ mkdir wcinput

**2****）在wcinput****文件下创建一个word.txt****文件**

[atguigu@hadoop102 hadoop-3.1.3]$ cd wcinput

**3****）编辑word.txt****文件**

[atguigu@hadoop102 wcinput]$ vim word.txt

Ø 在文件中输入如下内容

hadoop yarn

hadoop mapreduce

atguigu

atguigu

Ø 保存退出：:wq

**4****）回到Hadoop****目录/opt/module/hadoop-3.1.3**

**5****）执行程序**

[atguigu@hadoop102 hadoop-3.1.3]$ bin/hadoop jar share/hadoop/mapreduce/hadoop-mapreduce-examples-3.1.3.jar wordcount wcinput/ ./wcoutput

**6****）查看结果**

[atguigu@hadoop102 hadoop-3.1.3]$ cat wcoutput/part-r-00000

看到如下结果：

atguigu 2

hadoop 2

mapreduce    1

yarn  1



## 3.2 完全分布式运行模式（开发重点）

分析：

​    1）准备3台客户机（关闭防火墙、静态IP、主机名称）

​    2）安装JDK

​    3）配置环境变量

​    4）安装Hadoop

​    5）配置环境变量

**6****）配置集群**

**7****）单点启动**

​    **8****）配置ssh**

​    **9****）群起并测试集群**

## 3.2.2 编写集群分发脚本xsync





















