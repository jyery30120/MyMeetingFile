## 0818更新

把 num_train_steps 調高到15000  
用了一整個下午
跑出一個Eval results (程式如果跑成功就會有這段)

```shell
***** Eval results *****
global_step = 15000
loss = 0.38196152
masked_lm_accuracy = 0.9042763
masked_lm_loss = 0.38192594
next_sentence_accuracy = 1.0
next_sentence_loss = 3.5455356e-05
```
可以看到我們要的masked_lm_accuracy提升到90%  
給107字的句子(從天龍八部文本裡隨機挑一個段落)測試，從原本模型預測出40個錯字，降到24個   
另給234字句子測試，則從預測出76字降到58字  
可以得知這個方法可能是有效的

google_bert.ipynb 裡面是預訓練的範例  
keras_bert.ipynb 是套用模型及錯字偵測的小程式

未來努力方向：
1. 做NER，辨識錯字前先過濾掉專有名詞
2. 調整參數，找到更好的模型
3. 提高程式找錯字的效率

# Keras-bert 正文

暫時用Keras-BERT  
原因：Keras-BERT 有明確的load model方法，PyTorch暫時沒找到如何load自己訓練完的model的方法

Pretrain：   
跑Google Bert( https://github.com/google-research/bert )下面兩支程式   
應該要先把整包Bert程式碼下載下來才能跑下面兩支

```shell
python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

上面--input_file=./sample_text.txt \ 換成自己的檔案

```shell
python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
```

上面steps從20改到500 跑出來的結果

```shell
//init
 ***** Eval results *****
global_step = 20
loss = 2.3418965
masked_lm_accuracy = 0.58585525
masked_lm_loss = 2.0519152

//--num_train_steps=500 \
***** Eval results *****
global_step = 500
loss = 1.5109024
masked_lm_accuracy = 0.70651317
masked_lm_loss = 1.327778
```

以上是 masked_lm_accuracy 的變化  
然後套用到Keras bert

## keras bert

在我的資料夾裡的keras_bert.ipynb可以看到我改的程式跟我跑出來的結果  
基本就照著( https://github.com/CyberZHG/keras-bert/blob/master/demo/load_model/keras_bert_load_and_predict.ipynb )打

```shell
import os

pretrained_path = 'chinese_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

```
以上跑一次後再把pretrained_path換成自己的Model資料夾再跑一次做比較

然後...可以看到建議的字的部分有改變
但錯字率反而變更高

BERT 的模型：預測錯字數目: 40
自己訓練的模型：預測錯字數目: 42

列為未來改進方向

繼續努力方向：

1. 拉高epoch 提高 masked_lm_accuracy 但要避免overfitting
2. 透過自己的文本去fine-tune好的模型降低在目前文本的抓錯率
3. 針對詞做兩個以上的[MASK] 做錯詞辨識
4. keras-bert跟pytorch多方嘗試，並做比較
