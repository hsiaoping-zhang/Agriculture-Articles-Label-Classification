from mimetypes import init

from Dict import Dict

import pandas as pd
import numpy as np

'''
一個分組的工具將文章分組到 DataFrame 對應地方
'''

class Group:
    def __init__(self, currentFolder, cropDict):
        self.currentFolder = currentFolder
        self.df = None
        self.createDataFrame()
        self.cropDict = cropDict
        self.answerResult = []

    def createDataFrame(self):
        self.df = pd.read_csv("../train/KeyWords/02crop_list.csv", encoding='utf-8-sig', header=None)
        self.df["articles"] = pd.Series([[] for i in range(len(self.df))])

    # 根據作物分組
    def groupToOneGroup(self, fileNum):
        file = open(f"../{self.currentFolder}/data/{fileNum}.csv", "r")
        rows = file.readlines()

        # extract crop info
        cropRow = rows[0].strip("\n").split(",")
        # too many crops, ignore it
        if(len(cropRow) > 5):
            if("香蕉" in cropRow):
                cropRow = ["香蕉"]
            elif("水稻" in cropRow):
                cropRow = ["水稻"]

        for item in cropRow:
            if(item == ''):
                continue
            index = self.cropDict.dict[item]
            if(int(fileNum) not in self.df.iloc[index]["articles"]):
                self.df.iloc[index]["articles"].append(int(fileNum))