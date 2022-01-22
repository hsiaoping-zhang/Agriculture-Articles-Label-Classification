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
  - 斷詞前處理：jieba(0.42.1), scikit-learn(1.0.1)
  - 模型：tensorflow(0.1a3), keras(2.7.0)
  - 其它：pandas(1.3.4), numpy(1.21.4), plotly(5.4.0)
  - 訓練模型：Bidirectional LSTM
- model : `Bidirectional LSTM` 
- 額外資料集：[wiki model](https://github.com/music1353/Wikipedia-word2vec) (jieba_tw, stopwords)

## 流程
1. 資料前處理: `Data Preprocessing.ipynb`, `Word Frequency.ipynb`
> 斷詞及找出每篇文章的關鍵字組

2. 第一層過濾: `Main Filter.ipynb`
> 根據關鍵字組採用邏輯判斷子集計算

3. 模型: `Secondary Filter.ipynb`
> 使用 Bidirectional LSTM
