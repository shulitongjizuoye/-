# 文本情感识别模型2
使用了预训练的词袋模型tokenizer_lstm.pickle进行分词，预训练模型文件位置在..\model文件夹中。
 
### 代码注释:  
* try_function.py和try_function2.py两个程序都是用于训练时读取数据或抓取batch使用的函数。
* 首先应先运行try_train.py对有标签的20w条数据进行训练，视频对应01_train_label_data.mp4，保存的模型为GRU_1.h5文件。
* 然后运行try_get_label.py，将首次训练好的GRU_1.h5读入模型对没有标签的120w条数据进行分类，将分类的结果保存至..\RNN\training_label.txt中。
* 然后运行try_train2.py，将首次训练好的GRU_1.h5读入模型并读入没有标签的120W数据以及我们获取到的标签，对模型进行进一步训练，视频对应02_train_nolabel_data.mp4得到的模型保存至GRU_2.h5。
* 最后运行try_get_test_label.py，载入GRU_2.h5模型对testing.txt中的20w条文本数据进行分类，产生并保存model_2_test_label.csv，任务完成。