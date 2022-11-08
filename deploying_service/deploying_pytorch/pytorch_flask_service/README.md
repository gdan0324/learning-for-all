## flask搭建web服务

不适用生产环境：
1.pytorch模型未经优化
2.flask的WSGI不适合生产环境，需要配合一个高性能WSGI服务

## 模型优化

![image-20220903092428271](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220903092429.png)

提高推理速度，模型精度可能会下降，但也只是下降零点几个百分点。

pip install -r .\requirement.txt



同一局域网访问：

<img src="https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220903100924.png" alt="image-20220903100923528" style="zoom:67%;" />

192.168.1.6：5000

