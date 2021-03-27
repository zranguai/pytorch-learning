"""
pytorch可视化需要:
方法一：
pip install tensorboardX
1. 需要开启一个监听的进程

方法二：Visdom
1. pip install visdom
2. python -m visdom.server  (相当于开启了一个web服务器,web服务器会把数据渲染到网页上去)
    可能会遇到的问题: ERROR:root:Error 404 while downloading https://unpkg.com/layout-bin-packer@1.4.0

解决方法: install form source（从github的facebookresearch/visdom下载）
    步骤1: pip uninstall visdom
    步骤2: 官网下载源代码，之后cd进去目录(进去visdom-master)，之后运行pip install -e .
    步骤3： 退回用户目录后再python -m visdom.server
    步骤4：打开浏览器，输入他给的地址
"""


# 测试:
from visdom import Visdom
viz = Visdom()

"""
{Y的值，X的值} win可以理解为ID(还有一个id叫做env(默认使用main))   opts是额外的配置信息

对于非image还是numpy数据，image数据是tensor
"""
# viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
# viz.line([loss.item()], [global_step], win='train_loss', update='append')






