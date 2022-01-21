import re
import jieba
import jieba.analyse
import jieba.posseg as pseg 

from Dict import Dict

'''
文章斷詞工具
'''

class ArticleSegment:
    def __init__(self, currentFolder):
        self.currentFolder =  currentFolder
        self.stopWords = []
        self.dictList = []
        self.keyNameList = ["crop", "pest", "chem"]
        for item in self.keyNameList:
            itemDict = Dict(item)
            itemDict.loadDict(f"../train/KeyWords/02{item}_list.csv")
            self.dictList.append(itemDict)

        self.loadUserDict()
        self.loadStopwords("../stop_words.txt")

    def loadStopwords(self, filePath):
        stopwords_file = open(filePath, "r")
        stopwords = stopwords_file.readlines()
        self.stopwords = [item.strip("\n") for item in stopwords]

    def loadUserDict(self):
        jieba.set_dictionary('../dict.txt.big')
        jieba.load_userdict('../dict/chem_dict.txt')
        jieba.load_userdict('../dict/crop_dict.txt')
        jieba.load_userdict('../dict/pest_dict.txt')
        print("jieba dictionary OK.")

    # article segment to file
    def segment(self, filePath, fileNum):
        file = open(filePath, "r", encoding='utf-8-sig', errors='ignore')
        sentances = file.readlines()
        file.close()

        keywords = [[] for i in range(3)]
        text = self.removeSomeCharacter(sentances)
        words = pseg.cut(text)

        result = ""
        except_speech = ["m", "p", "d", "c", "eng"]
        for word, flag in words:
            # only extract a word with length >= 2 and part of speech with v, n, t, and key words!!
            if(word not in self.stopwords and len(word) > 1 and flag not in except_speech):
                result += (word + " ")

            # extract keywords
            if(flag not in self.keyNameList):
                continue
            index = self.keyNameList.index(flag)
            if(word not in keywords[index]):
                if(flag in self.keyNameList):
                    i = self.dictList[index].dict[word] # convert to represent word
                    word = self.dictList[index].list[i]
                    
                if(word in keywords[index]):  # 再次檢查是否重複
                    continue
                keywords[index].append(word)

            self.writeToSegmentResultFile(fileNum, result)
            self.writeToKeywordsFile(fileNum, keywords)
            

    def writeToSegmentResultFile(self, fileNum, result):
        # record to file with article segment
        file = open(f"../{self.currentFolder}/ArticleSegment/{fileNum}.txt", "w")  # 放入 /ArticleSegment 資料夾
        file.write(f"{result}\n")
        file.close()

    def writeToKeywordsFile(self, fileNum, keywords):
        # write to file with keywords
        # print(f"file:  {fileNum}")
        file = open(f"../{self.currentFolder}/data/" + str(fileNum) + ".csv", "w")  # 放入 /data 資料夾
        # print(keywords)

        for key_list in keywords:
            row = ""
            for key in key_list:
                row += (key + ",")
            row = row.strip(",") + "\n"
            file.write(row)

        file.close()
        # print("- - -")

    def removeSomeCharacter(self, sentances):
        text = ""
        for item in sentances:
            replace = re.sub(r'[0-9]+|[%]|[-]|[a-z]','', item.replace("\x7f", ""))
            text += replace
        return text