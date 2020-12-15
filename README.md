## 论文:Slot-Gated Modeling for Joint Slot Filling and Intent Prediction  

--------------------------------------------------------------------------------------------------------
## 数据集:

​        dstc8

​        注：论文中使用的数据集应该是是atis和snips，但是我做实验用的dstc8数据集。

​	dstc8数据集[点这](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)

​     atis数据集[点这](https://github.com/howl-anderson/ATIS_dataset)

--------------------------------------------------------------------------------------------------------

## 需要安装的包

```
pip install torch
pip install sklearn
pip install numpy
都是一些常见包，剩下的不举例了
```

## 代码

```
main.py
#模型的入口
builddataset.py
#模型的数据处理
slotfillingandintentdetermination.py
#模型
data  #数据集
saved_models #保存的模型
```

## dstc8实验结果

intent_acc:54.701

slot_f1:29.327