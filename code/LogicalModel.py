from json import load, tool
import pandas as pd

from Tool import getFileRows, splitDiseaseAndPest, extractChem, countChemOverlapNum

'''
第一階段方法的 Model
'''

class LogicalModel:
    def __init__(self, currentFolder, df, output):
        self.currentFolder = currentFolder
        self.realAnswer = {}
        self.realAnswerNum = 0
        self.loadAnswer(currentFolder)
        self.guessAnswer = []
        self.df = df
        self.outputFilePath = output

    # 載入答案作為 dictionary
    def loadAnswer(self, folder):
        if(folder != "train"):
            return
        answerFile = open("../TrainLabel.csv", "r")
        answerLines = answerFile.readlines()

        total = 0
        for i in range(1, len(answerLines)):
            items = answerLines[i].strip("\n").split(",")
            test, reference = int(items[0]), int(items[1])
            total += 1
            
            if(test in self.realAnswer):
                self.realAnswer[test].append(reference)
            else:
                self.realAnswer[test] = [reference]     
        self.realAnswerNum = total
        # print("total answer:", total)

    # 加到作答答案裡面
    def addToAnswerList(self, item):
        if(item not in self.guessAnswer):
            self.guessAnswer.append(item)

    # 兩篇文章是否相關
    def isArticlesRelated(self, file1, file2):
        rows1 = getFileRows(self.currentFolder, file1)
        diseaseAndPest1 = [] if rows1[1] == "\n" else rows1[1].strip("\n").split(",")
        chem1 = len(extractChem(rows1[2]))

        rows2 = getFileRows(self.currentFolder, file2)
        diseaseAndPest2 = rows2[1].strip("\n").split(",")
        chem2 = len(extractChem(rows2[2]))

        chemOverlap = countChemOverlapNum(rows1[2], rows2[2])

        diseases1, pest1 = splitDiseaseAndPest(diseaseAndPest1)
        diseases2, pest2 = splitDiseaseAndPest(diseaseAndPest2)

        # 判斷疫病主題：如果疫病一樣，則代表有關聯
        isDisease = False
        DiseaseCount = 0
        for disease in diseases1:
            if(disease in diseases2):
                isDisease = True
                DiseaseCount += 1

        # 判斷害蟲: 如果疫病+害蟲有重疊的話就算有關聯
        if(isDisease and DiseaseCount == len(diseases1)):
            count = 0
            for bug in pest1:
                if(bug in pest2):
                    count += 1    
            if(count != 0 and count >= len(pest1)/2 and len(pest1) != 0):
                return True
                
            # 沒有害蟲
            elif(count == 0):
                # 為了限縮可能的文章組合，針對 chem 關鍵字類別進行篩選取捨
                if((chem1 != 0 and chem2 != 0) and (chemOverlap >= chem1/2)):
                    return True
                elif(chem1 == 0 and chem2 == 0):
                    return True
        
        # 沒有疫病的比較
        if(len(diseases1) == 0):
            return self.compareInNoDiseaseCondition(pest1, pest2, chemOverlap, chem1)

        return False

    # 沒有疫病的狀況下的比較: 害蟲指示需要相同
    def compareInNoDiseaseCondition(self, pest1, pest2, chemOverlap, chem1):
        overlapCount = 0
        for bug in pest1:
            if(bug in pest2):
                overlapCount += 1

        # 害蟲關鍵字有交集
        if(overlapCount >= len(pest1)/2 and len(pest2) != 0 and chemOverlap >= chem1/2): 
            return True

        return False

    def compareAllCombination(self):
        for index in range(len(self.df)):
            articles = self.df.iloc[index]["articles"]
            # 提及同個作物的文章相互比對
            for test1 in articles:
                for test2 in articles:
                    if(test1 == test2):
                        continue
                    ans = self.isArticlesRelated(test1, test2)
                    if(ans):  # if True
                        self.addToAnswerList([test1, test2])
        print("- - - -")
        print(f"[ First Stage in {self.currentFolder} ] all possible candidates:", len(self.guessAnswer))

        if(self.currentFolder == "train"):
            self.createTrainFile()
            self.checkAnswer
        else:
            self.writeFileOfCandidates()

    # record candidates to file
    def writeFileOfCandidates(self):
        file = open(self.outputFilePath, "w")
        file.write("Test,Reference\n")
        for item in self.guessAnswer:
            file.write(f"{item[0]},{item[1]}\n")
        file.close()

    def createTrainFile(self):
        trainLabelFile = open("../train/all-candidate.csv", "w") # record related, unrelated as training data
        trainLabelFile.write("Test,Reference,Label\n")
        for [index, article] in self.guessAnswer:
            if(index in self.realAnswer and article in self.realAnswer[index]):
                trainLabelFile.write(f"{index}, {article}, 1\n")  # related
            else:
                trainLabelFile.write(f"{index}, {article}, 0\n")  # unrelated
        trainLabelFile.close()

    # 跟正確答案進行比較
    def checkAnswer(self, isPrint):
        if(self.currentFolder != "train"):
            return
        reverse, true, error = 0, 0, 0

        for index in self.realAnswer:
            articles = self.realAnswer[index]
            for article in articles:
                if([index, article] in self.guessAnswer):
                    true += 1
                else:
                    if([article, index] in self.guessAnswer):
                        if(isPrint):
                            print(f"[{index}, {article}] reverse exist.")
                        reverse += 1
                    else:
                        if(isPrint):
                            print(f"[{index}, {article}] not in predict answer.")   
                        error += 1

        precision = true / len(self.guessAnswer)
        recall = true / (self.realAnswerNum)
        score = 2 * (precision * recall) / (precision + recall)
        print(f"precision: {round(precision, 2)} ({true} / {len(self.guessAnswer)})")
        print(f"recall: {round(recall, 2)} ({true} / {self.realAnswerNum})")
        print(f"score: {round(score, 2)}")
        print(f"reverse exist: {reverse} | not found: {error}")
        