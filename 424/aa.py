# coding=utf-8

from PIL import Image

import os.path

import glob


# python 函数

# 功 能：将一张 jpg文件转pgm格式文件

# 参 数：jpg_file : 欲转换的jpg文件名

#              pgm_dir  : 存放 pgm 文件的目录

def jpg2pgm(jpg_file, pgm_dir):
    # 首先打开jpg文件

    jpg = Image.open(jpg_file)

    # resize to 200 * 250 , 双线性插值

    jpg = jpg.resize((200, 250), Image.BILINEAR)

    # 调用 python 函数 os.path.join , os.path.splitext , os.path.basename ，产生目标pgm文件名

    name = (str)(os.path.join(pgm_dir, os.path.splitext(os.path.basename(jpg_file))[0])) + ".pgm"

    # 创建目标pgm 文件

    jpg.save(name)


# 将所有的jpg文件放在当前工作目录，或者 cd {存放jpg文件的目录}

for jpg_file in glob.glob("./*.jpg"):
    jpg2pgm(jpg_file, "tools")


