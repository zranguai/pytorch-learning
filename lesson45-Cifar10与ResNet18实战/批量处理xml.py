#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
https://blog.csdn.net/gusui7202/article/details/83239142
qhy。
"""
"""
import os
import xml.etree.ElementTree as ET

origin_ann_dir = r'E:\dataProcess\smoke_data\1_yuanlai'  # 设置原始标签路径为 Annos
new_ann_dir = r'E:\dataProcess\smoke_data\2_gaibian'  # 设置新标签路径 Annotations
for dirpaths, dirnames, filenames in os.walk(origin_ann_dir):  # os.walk游走遍历目录名
    for filename in filenames:
        print(filename)
        if os.path.isfile(r'%s%s' % (origin_ann_dir, filename)):  # 获取原始xml文件绝对路径，isfile()检测是否为文件 isdir检测是否为目录
            origin_ann_path = os.path.join(r'%s%s' % (origin_ann_dir, filename))  # 如果是，获取绝对路径（重复代码）
            new_ann_path = os.path.join(r'%s%s' % (new_ann_dir, filename))  #
            tree = ET.parse(origin_ann_path)  # ET是一个xml文件解析库，ET.parse（）打开xml文件。parse--"解析"
            root = tree.getroot()  # 获取根节点
            for object in root.findall('object'):  # 找到根节点下所有“object”节点
                name = str(object.find(
                    'name').text)  # 找到object节点下name子节点的值（字符串），判断:如果不是列出的，（这里可以用in对保留列表成员进行审查），则移除该object节点及其所有子节点。
                if not (name in ["fire"]):    # 3-18这里做改变
                    root.remove(object)
            flag = 0  # 清楚非保留完成-标志位0
            tree.write(new_ann_path)  # tree为文件，write写入新的文件中。
            for object in root.findall('object'):  # 找到根节点下所有子节点
                name = str(object.find('name').text)  # 找到子节点中name变量，判断：如果每一个都是要保留的，则标志位变1，这是一个审查。
                if (name in ["fire"]):    # 3-18这里做改变
                    flag = 1
            if (flag == 0):
                os.remove(new_ann_path)  # 所有不满足审查：有多余object，则用os.remove(filepath)删除指定文件。
# 注意:改两个地方，分别两个有注释的地方，一个if,一个if not
"""
#  批量移除xml标注中的某一个类别标签
import xml.etree.cElementTree as ET
import os


path_root = [r'E:\dataProcess\smoke_data\1_yuanlai']

CLASSES = [
    "yanwu"]
for anno_path in path_root:
    xml_list = os.listdir(anno_path)
    for axml in xml_list:
        path_xml = os.path.join(anno_path, axml)
        tree = ET.parse(path_xml)
        root = tree.getroot()

        for child in root.findall('object'):
            name = child.find('name').text
            if not name in CLASSES:
                root.remove(child)

        tree.write(os.path.join(r'E:\dataProcess\smoke_data\2_gaibian', axml))
