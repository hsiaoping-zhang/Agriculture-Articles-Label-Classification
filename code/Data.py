import pandas as pd

from Tool import getArticleSegment

'''
在第二階段方法時的輸入 data 物件(包含 csv 內容及 DataFrame 狀態)
'''

class Data:
    def __init__(self, name, filePath):
        self.name = name
        self.data = self.readLabelFile(filePath)  # csv rows
        self.df = self.constructDataFrame()       # dataframe
        
    def readLabelFile(self, filePath):
        file = open(filePath, "r")
        result = file.readlines()[1:]
        result = [item.strip("\n") for item in result]
        return result

    def constructDataFrame(self):
        print("construct", self.name, "data...")
        sentance1, sentance2, label = [], [], []
    
        # 兩兩文章組合各自讀檔
        for itemString in self.data:
            items = itemString.split(",")
            item1, item2 = int(items[0]), int(items[1])
            sentance1.append(getArticleSegment(self.name, item1))
            sentance2.append(getArticleSegment(self.name, item2))
            if(self.name == "train"):
                label.append(int(items[2]))
        
        # 文章 1 與文章 2 放在同一個 row
        df = pd.DataFrame()
        df["sentance_1"] = sentance1
        df["sentance_2"] = sentance2

        if(self.name == "train"):
            df["label"] = label

        return df