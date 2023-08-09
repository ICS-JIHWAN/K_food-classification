
"""
    == Looking for K-food dataset ==

    loot directory path : '/storage/jhchoi/kfood'

    dir - kimchi
            | - kimchi   : 1000
            | - kkakdugi : 1000

    dir - noodle
            | - banquet_noodles : 1000
            | - kalguksu        : 1000
            | - ramen           : 1000

    dir - pancake
            | - gyeranmari      : 1002
            | - pajeon          : 1001
            | - pumpkin_pancake : 1000

    dir - rice
            | - bibimbap : 1000
            | - gimbap   : 1000

    dir - soup
            | - egg_soup         : 1001
            | - sea_mustard_soup : 1000
            | - yukgaejang       : 1000

    png file : cv2.imread(file path)
    other    : plt.imread(file path)

    If you open the image to open-cv library,
        image channels consist of BGR.
    So, to use the image which was opened with open-cv,
        you must use cv2.cvtColor in order to BGR to RGB.

"""

import os
import glob
import matplotlib.pyplot as plt
import cv2

root_path = '/storage/jhchoi/kfood'
dir = glob.glob(os.path.join(root_path, '*/*/*'))

jpg = []    # 12684
png = []    # 49
JPG = []    # 243
jpeg = []   # 15
other = []  # 13

four_channel = []
#
#
# for d in dir:
#     if '.jpg' == d[-4:]:
#         jpg.append(d)
#         i = plt.imread(d)
#         if i.shape[-1] != 3:
#             four_channel.append(d)
#
#     elif '.png' == d[-4:]:
#         png.append(d)
#         i = cv2.imread(d, cv2.IMREAD_COLOR)
#         if i.shape[-1] != 3:
#             four_channel.append(d)
#
#     elif '.JPG' == d[-4:]:
#         JPG.append(d)
#         i = plt.imread(d)
#         if i.shape[-1] != 3:
#             four_channel.append(d)
#
#     elif '.PNG' == d[-4:]:
#         PNG.append(d)
#
#     elif '.jpeg' == d[-5:]:
#         jpeg.append(d)
#         i = plt.imread(d)
#         if i.shape[-1] != 3:
#             four_channel.append(d)
#
#     else:
#         other.append(d)
#         i = plt.imread(d)
#         if i.shape[-1] != 3:
#             four_channel.append(d)
