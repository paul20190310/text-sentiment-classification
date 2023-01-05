# 資料集
本系統採用訓練資料集為「Emotion資料集」，為Twitter英文訊息，並帶有六種基本情緒標籤，分別生氣(anger)、恐懼(fear)、愛(love)、歡愉(joy)、傷心(sadness)、以及驚訝(surprise)，更多資訊詳見：<a href="https://huggingface.co/datasets/emotion">https://huggingface.co/datasets/emotion</a>
## 使用
### Non-IID資料分割
#### 數量導向分割
將資料分割成不同數量，'-r'為分割比例、'-f'為生成檔案名稱、'-s'為種子。下例為將資料分割成三個檔案，資料占比分別為20%、30%、及50%
```
$ python partition_dataset.py emotion.csv -r 0.2 0.3 0.5 -f emotion_2.csv emotion_3.csv emotion_5.csv -s 2023
```
#### 標籤導向分割
將同種標籤資料分割成不同檔案，加入'-l'指令，此時'-r'為分割標籤數量、'-f'及'-s'同數量導向。下例為將資料分割成三個檔案，資料所占標籤種類分別為1種、2種、及3種。
```
$ python partition_dataset.py emotion.csv -r 1 2 3 -f emotion_1.csv emotion_2.csv emotion_3.csv -s 2023 -l
```
### 訓練集資料切割
切割原始訓練資料集為三部分，訓練、驗證、及測試集(指令取消'-t'可只產生訓練及驗證集)
```
$ python split_dataset.py emotion.csv -t
```
