from json import load

'''
不同關鍵字會建立不同內容的 dictionary
'''

class Dict:
    def __init__(self, name):
        self.dictName = name
        self.dict = {}  # 同義詞對應同一索引
        self.list = []  # 所有詞的 list

    def loadDict(self, filePath):
        file = open(filePath, "r", encoding='utf-8-sig')
        # only get content with .readlines() method, the program can read the abnormal character
        rows = file.readlines()  
        file.close()
        
        for i in range(len(rows)):
            row = rows[i].strip("\n").strip(",").replace("\x7f", "").split(",")
            self.list.append(row[0])  # use first item as represent
            for item in row:
                self.dict[item] = i  # the same meaning words with the same index nuber