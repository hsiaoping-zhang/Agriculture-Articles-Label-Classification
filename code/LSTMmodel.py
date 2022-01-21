import imp
from Tool import getArticleSegment, getMainCrop
from Data import Data

import pandas as pd
import numpy as np
from gensim.models import word2vec
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, LSTM, concatenate, Dense, Bidirectional, GRU, Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model

NUM_CLASSES = 2          # related / unrelated

class LSTMmodel:
    def __init__(self, trainFile, testFile, output, maxWords, maxSequence, embeddingDim, lstmUnit, batchSize, epoch):
        self.train = Data("train", trainFile)  # train   folder
        self.test = Data("private", testFile)  # private folder
        
        self.word2vecModel = word2vec.Word2Vec.load("../wiki_model.bin") # pretrain-model path
        self.word2idx = {}
        self.outputFile = output
        print("[ Second Stage ]", end=" ")

        self.MAX_NUM_WORDS = maxWords
        self.MAX_SEQUENCE_LENGTH = maxSequence  # 一篇文章最長有幾個字詞
        self.NUM_EMBEDDING_DIM = embeddingDim   # 一個詞向量的維度
        self.NUM_LSTM_UNITS = lstmUnit          # LSTM 輸出的向量維度
        self.BATCH_SIZE = batchSize
        self.NUM_EPOCHS = epoch

    # 找到對應的 embedding matrix index
    def textToEmbeddingInedex(self, textList):
        new_corpus = []
        for doc in textList:
            new_doc = []
            for word in doc:
                try:
                    new_doc.append(self.word2idx[word])
                except:
                    new_doc.append(0)  # not in model, replace with 0
            new_corpus.append(new_doc)

        return new_corpus

    def createEmbedding(self):
        print("create embedding...")
        embedding_matrix = np.zeros((len(list(self.word2vecModel.wv.index_to_key)) + 1, self.word2vecModel.vector_size))
        vocab_list = [(word, self.word2vecModel.wv[word]) for word in list(self.word2vecModel.wv.key_to_index.keys())]

        i = 0
        for vocab in enumerate(vocab_list):
            word = vocab
            word, vec = vocab_list[i][0], vocab_list[i][1]

            embedding_matrix[i + 1] = vec
            self.word2idx[word] = i + 1
            i += 1

        embedding_layer = layers.Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False)

        return embedding_layer

    # 文章轉成數字序列
    def createInputData(self, textList):
        data = self.textToEmbeddingInedex(textList)
        data = tf.keras.preprocessing.sequence.pad_sequences(data, maxlen = self.MAX_SEQUENCE_LENGTH, padding='post')
        return data

    def createModel(self):

        # 分別定義 2 個文章 q1 & q2 為模型輸入兩個文章都是一個長度為 100 的數字序列
        q1_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        q2_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,),dtype='int32')


        # 詞嵌入層：經過詞嵌入層的轉換，兩個文章都變成一個詞向量的序列，而每個詞向量的維度為 256
        embedding_layer = self.createEmbedding()
        q1_embedded = embedding_layer(q1_input)
        q2_embedded = embedding_layer(q2_input)

        # LSTM 層：兩個文章經過此層後為一個 128 維度向量
        shared_lstm = Bidirectional(LSTM(self.NUM_LSTM_UNITS, dropout=0.1))
        q1_output = shared_lstm(q1_embedded)
        q2_output = shared_lstm(q2_embedded)

        # 串接層將兩個文章的結果串接單一向量方便跟全連結層相連
        merged = concatenate(
            [q1_output, q2_output], 
            axis=-1)

        outer_dense = Dense(32, activation="relu")

        # 全連接層搭配 Softmax Activation 可以回傳 2 個文章屬於各類別的可能機率
        dense =  Dense(units=NUM_CLASSES, activation='softmax')

        # predictions = dense(merged)
        predictions = dense(outer_dense(merged))

        # 模型就是將數字序列的輸入轉換成 2 個分類的機率
        model = Model(
            inputs=[q1_input, q2_input], 
            outputs=predictions)
        return model

    def trainModelAndCheck(self):
        print("train model...")
        model = self.createModel()
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        VALIDATION_RATIO = 0.2
        RANDOM_STATE = 40

        x_train_q1 = self.createInputData(self.train.df["sentance_1"])
        x_train_q2 = self.createInputData(self.train.df["sentance_2"])
        y_train = self.train.df["label"]

        # 訓練集切塊
        x_train_q1, x_val_q1, x_train_q2, x_val_q2, y_train, y_val = \
            train_test_split(
                x_train_q1, x_train_q2, y_train, 
                test_size=VALIDATION_RATIO, 
                random_state=RANDOM_STATE
            )
        # 將 label 轉換成 related / unrelated
        y_train = tf.keras.utils.to_categorical(y_train)
        y_val = tf.keras.utils.to_categorical(y_val)

        # 實際訓練模型
        history = model.fit(
            # 輸入是兩個長度為 100 的數字序列
            x=[x_train_q1, x_train_q2], 
            y=y_train,
            batch_size=self.BATCH_SIZE,
            epochs=self.NUM_EPOCHS,
            # 每個 epoch 完後計算驗證資料集上的 Loss 以及準確度
            validation_data=(
                [x_val_q1, x_val_q2], 
                y_val
            ),
            # 每個 epoch 隨機調整訓練資料集裡頭的數據以讓訓練過程更穩定
            shuffle=True
        )
        
        trainPredictions = model.predict([x_train_q1, x_train_q2])
        self.checkAnswer(trainPredictions)

        return model
    
    def applyModel(self, model):
        print("apply model...")

        x_test_q1 = self.createInputData(self.test.df["sentance_1"])
        x_test_q2 = self.createInputData(self.test.df["sentance_2"])

        # 利用已訓練的模型做預測
        testPredictions = model.predict([x_test_q1, x_test_q2])
        self.writeAnswer(testPredictions, 0.7, self.outputFile)

    # check training data set's answer
    def checkAnswer(self, prediction):
        errors = [0 for i in range(3)]  # pred=1 ans=0
        losses = [0 for i in range(3)]  # pred=0 ans=1

        # prediction
        pred_true = [0 for i in range(3)]
        pred_false = [0 for i in range(3)]

        # answer
        ans_true = [0 for i in range(3)]
        ans_false = [0 for i in range(3)]

        # article number count
        crops = [0 for i in range(3)]

        true_11 = [0 for i in range(3)]
        true_00 = [0 for i in range(3)]

        threshold = 0.7

        for i in range(len(prediction)):
            row = self.train.data[i].split(",")
            item1, item2 = int(row[0]), int(row[1])
            
            crop = getMainCrop("train", item1)
            # add to crop list
            index = 0 if (crop == "水稻") else 1 if("蕉" in crop) else 2
            crops[index] += 1
            
            pred = prediction[i]
            predLabel = 0 if(pred[0] > threshold) else 1
            ansLabel = self.train.df.iloc[i]["label"]
            
            if(predLabel == 0 and ansLabel == 1):
                pred_false[index] += 1
                ans_true[index] += 1
                losses[index] += 1
                # print(f"unlike: [{item1}, {item2}]", round(pred[0], 2))
                    
            elif(predLabel == 1 and ansLabel == 1):
                pred_true[index] += 1
                ans_true[index] += 1
                true_11[index] += 1
                # print(f"true related: [{item1}, {item2}] {main_crop(item1)} -> {main_crop(item2)}")

            elif(predLabel == 1 and ansLabel == 0):
                pred_true[index] += 1
                ans_false[index] += 1
                errors[index] += 1
                # print(f"like: [{item1}, {item2}] {round(pred[0], 2)}")
                
            else:
                pred_false[index] += 1
                ans_false[index] += 1
                true_00[index] += 1
                # print(f"true unrelated: [{item1}, {item2}] {main_crop(item1)} -> {main_crop(item2)}")

        right, total_guess, total_related = 0, 0, 0
        for i in range(3):
            right += true_11[i]
            total_guess += pred_true[i]
            total_related += ans_true[i]
            
        print("Threshold:", threshold)
        print("[ format: guess/answer ]")
        print("%-10s | %-10s | %-10s | %-10s | %-10s" % ("crop", "00", "01(unlike)", "10(like)", "11"))
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        names = ["rice", "banana", "others"]
        for i in range(3):
            print("%-10s | %-10d | %-10d | %-10d | %-10d" % (names[i], true_00[i], losses[i], errors[i], true_11[i]))
        print()

        precision = right / total_guess
        recall = right / total_related
        base = 0.85  # 假設上一層的 recall 最多只能達到 0.85
        print(f"precision: {round(precision, 2)} | total guess: {total_guess} | right: {right}")
        print(f"recall: {round(recall, 2)} | total related: {total_related}")

        recall = recall * base
        score = 2*(precision * recall) / (precision + recall)

        print(f"socre: {round(score, 2)} | recall: {round(recall, 2)}")
        print(f"total candidate: {len(self.train.df)}")

    def writeAnswer(self, prediction, threshold, outputPath):
        # write to file
        file = open(outputPath, "w")
        file.write("Test,Reference\n")
        test, ref = [], []
        pred_true, pred_false = 0, 0

        for i in range(len(prediction)):
            pred = prediction[i]
            
            predLabel = 0 if(pred[0] > threshold) else 1
            row = self.test.data[i].split(",")

            item1, item2 = int(row[0]), int(row[1])
            
            if(predLabel == 1):
                file.write(f"{item1}, {item2}\n")
                test.append(item1)
                ref.append(item2)
                pred_true += 1
                
            else:
                pred_false += 1
        file.close()
        print(f"related: {pred_true} | unrelated: {pred_false} | ratio: {round(pred_true/len(prediction), 2)}")
        print("total prediction:", len(prediction))