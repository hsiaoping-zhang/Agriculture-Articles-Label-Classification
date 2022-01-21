import re
import jieba
import jieba.analyse
import jieba.posseg as pseg 

from os import listdir
from os.path import isfile, isdir, join

from ArticleSegment import ArticleSegment
from Group import Group
from LogicalModel import LogicalModel
from LSTMmodel import LSTMmodel

print("start...")

segment = ArticleSegment("train")  # 用來做文章斷詞的相關工具物件

'''
文章斷詞及提取關鍵字處理

(如果已經斷過詞，這段可以註解起來，只執行一二階段方法就好)
'''
# folders = ["train", "private"]
# for folder in folders:
#     # all file in directory
#     mypath = f"../{folder}/data{folder.capitalize()}Complete"
#     files = listdir(mypath)
#     for f in files:
#         fullpath = join(mypath, f)
#         if isfile(fullpath):
#             fileNum = f.split(".")[0]
#             segment.segment(fullpath, fileNum)

'''
第一階段方法： 分組進行比較
'''
folders = ["train", "private"]
for folder in folders:
    # all file in directory
    mypath = f"../{folder}/data{folder.capitalize()}Complete"
    files = listdir(mypath)
    grouping = Group(folder, segment.dictList[0])  # crop dict in segment
    for f in files:
        fullpath = join(mypath, f)
        if isfile(fullpath):
            fileNum = f.split(".")[0]
            grouping.groupToOneGroup(fileNum)  # 分組到 DataFrame 某個欄位裡
            
    outputPath = f"../{folder}/all-candidate.csv"  # 第一階段方法輸出的檔案名稱
    firstModel = LogicalModel(folder, grouping.df, outputPath)  # df 指派給 model 進行第一階段方法
    firstModel.compareAllCombination()

    # # # for training data set # # #
    if(folder == "train"):
        firstModel.checkAnswer(False)
print("- - - -")

'''
第二階段方法： Bi-LSTM
'''
lstm = LSTMmodel(trainFile="../train/all-candidate.csv", 
                testFile="../private/all-candidate.csv", 
                output="../private/result-0121.csv",
                maxWords=15000,   # 一篇文章最長有幾個字詞
                maxSequence=100,  # 一個詞向量的維度
                embeddingDim=256, 
                lstmUnit=128,     # LSTM 輸出的向量維度
                batchSize=100,   # 決定一次要放多少成對標題給模型訓練  70 (for the best score)
                epoch=20)          # 決定模型要看整個訓練資料集幾遍  20 (for the best score)
model = lstm.trainModelAndCheck()
lstm.applyModel(model=model)

print("- - - finish - - -")
