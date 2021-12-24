# Agriculture Articles Label Classification
AI CUP 2021 競賽: 農業文章文字標註及辨識

\#更新中

內容：行政院農委會開放資料平台下載下來的「植物疫情」相關文字資料，以及「疫情通報」兩兩文章相似程度的標註結果，以人工智慧建立相似度的辨識分析模組。

- Training Data Set：560篇 (共560 x 559種組合；Yes or No)
- Test Data Set (Public)：421篇 (共421 x 420種組合；Yes or No)
- Test Data Set (Private)：420篇 (共420 x 419種組合；Yes or No

## 環境
- language : `python 3.8`
- package :
  -  `jieba`
  -  `scikit-learn`
  -  `tensorflow`
  -  `keras`
  -  `pandas`
  -  `numpy`
  -  `plotly`
- model : `Bidirectional LSTM` 

## 流程
1. 斷詞: `Data Preprocessing.ipynb`, `Word Frequency.ipynb`
2. 過濾: `Main Filter.ipynb`
3. 模型: `Secondary Filter.ipynb`


