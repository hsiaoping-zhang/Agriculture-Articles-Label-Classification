'''
一些在不同 class 會用到的 function
但不牽涉 class 元素
'''

# 把疾病跟害蟲的關鍵字分開(同屬 pest)
def splitDiseaseAndPest(items):
    disease, pest = [], []
    for item in items:
        if("病" in item):
            disease.append(item)
        else:
            pest.append(item)
    return disease, pest

# 把化學藥劑關鍵字提取出來
def extractChem(row):
    row = row.replace("\n", "")
    if(row == ""):
        return []
    else:
        return row.split(",")


# 比較兩篇文章的的化學藥劑相同數
def countChemOverlapNum(row1, row2):
    item1 = extractChem(row1)
    item2 = extractChem(row2)
    count = 0
    for item in item1:
        if item in item2:
            count += 1
    return count  

# 獲得某一篇文章的關鍵字群
def getFileRows(currentFolder, fileNum):
    file = open(f"../{currentFolder}/data/{fileNum}.csv", "r")
    rows = file.readlines()
    file.close() 
    return rows

# 獲得某一篇文章的斷詞結果
def getArticleSegment(currentFolder, fileNum):
    file = open(f"../{currentFolder}/ArticleSegment/{fileNum}.txt", "r")
    sentances = file.readlines()[0].strip("\n")
    file.close()
    return sentances

# 獲得某一篇文章的主要作物名稱
def getMainCrop(currentFolder, fileNum):
    rows = getFileRows(currentFolder, fileNum)
    item = rows[0].strip("\n").split(",")[0]
    # row2 = [] if rows[1] == "\n" else rows[1].strip("\n").split(",")

    return item