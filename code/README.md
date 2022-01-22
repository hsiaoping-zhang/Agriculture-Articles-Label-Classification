README
===

### 架構
- `main.py` : 主程式
- `ArticleSegment.py` : 斷詞相關
- `Group.py` : 第一階段方法(作物分組)
- `Dict.py` : 關鍵字字典 class
- `LogicalModel.py` : 第一階段方法(選出候選組合)
- `LSTMmodel.py` : 第二階段方法
- `Data.py` : 第二階段 input data type
- `Tool.py` : 其它 function


### 資料前處理
使用主辦單位提供的 `02crop_list.xlsx`、`02pest_list.xlsx`、`02chem_list.xlsx` 關鍵字字典參考進行輔助斷詞，並透過斷詞後的結果進行關鍵字提取，先將提及相同作物的文章聚集在一起，之後再比對具備疫病、害蟲關鍵字的文章組合，先行儲存可能的組合以作為後續訓練的依據。
除此之外，也對整體文章進行字詞數計數，排除次數過多與過少的詞，將處理過後新的斷詞結果作為後續模型輸入。

### 階段方法

#### 第一階段方法 : 流程判斷

先將提及相同作物的文章們進行分組，接著除了一般的邏輯判斷外，因為注意到文章組合具備非對稱性的特點，因此在化學藥劑及害蟲數量的計算比較上，都是以 A 文章作為比較基準。  

<img src="https://i.imgur.com/yM7udIu.png" width="500"/>


#### 第二階段方法 : Bidirectional LSTM

針對第一層的 embedding 層輸入，使用網路上別人已訓練好的模型 (使用維基百科文章資料進行訓練) 作為詞向量參考，在進行詞向量的轉換後，兩篇文章會各自進入 Bidirectional LSTM，之後再用 concatenate 層串起兩個LSTM，最後則使用 dense 層，輸出維度為 32，為輸出相關與不相關的機率個數。

下圖為模型輸出與輸入參數相關資訊：

<img src="https://i.imgur.com/gekAadP.png"  width="500"/>
<!-- ![](https://i.imgur.com/gekAadP.png) -->


### 訓練方式
調整 `batch size` / `epoch` 數字進行測試訓練，並對 

- `optimizer` (rmsprop, adam)
- `loss` (binary_crossentropy,categorical_crossentropy)
- `metrics` (binary_accuracy, accuracy) 

進行不同參數的替換調整，期望在訓練結果底下能將 prediction=1, answer=0 及 prediction=0, answer=1 的誤差盡量減少。

(如下圖在程式當中的檢視，unlike 為模型判斷為不相關，但實際上為相關；like 則為模型判斷為相關，但實際上為不相關。0.7 為 threshold，如果不相關的機率大於 0.7 才將此文章組合歸類為不相關)

<!-- ![](https://i.imgur.com/Rhtfv7S.png) -->
<img src="https://i.imgur.com/Rhtfv7S.png"  width="400"/>

### 外部資源與參考文獻
music1353 (2018), [維基百科詞向量 model](https://github.com/music1353/Wikipedia-word2vec)

LeeMeng (2018), [進入 NLP 世界的最佳橋樑：寫給所有人的自然語言處理與深度學習入門指南](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html)
