# Introducton
该程序是Traffic Forecasting的实验框架。

### 主要函数介绍
1. exp.py 
主函数. 在参数设定正确的调节下,在终端使用 "python exp.py" 运行实验 或者使用 "python -u exp.py >> logname.log &", 以将程序挂起, 并将终端输出保存到 logname.log 文件中.

2. analyse_tools.py
结果统计函数. 当 exp.py 运行完毕时, 将在路径 "args: root_path + exp_info_path" 中保存实验数据结果class对象, 运行该程序将自动将该对象解析并保存为 Excel 文件. 该文件的 Page 1 为 train, vali, test 三个数据集的所有 Epoch 的 MAE, Page 2 则是对最佳指标的统计.

这些统计数据包含 最佳Train/vali/test的Loss, 最佳 vali loss对应的epoch值,最佳vali loss epoch 对应的 MAE, MAPE, RMSE均值以及12步结果.
Ps: 目前程序无法同时处理不同 epoch 长度的实验结果文件, 请确认 "path = 'xxx/results/dict/'" 中所有结果文件拥有相同的 epoch.

3. generate_data
数据生成程序. 其中
generate_csv_file: h5/npz -> csv, 将 metr-la 和 pems-bay 的 h5 文件以及 pems-04/08 的 npz 文件转化为 带有日期信息的 csv文件.  
generate_torch_data_set: csv -> torch dataloader, 将 csv 文件转换为pytoch可用的Dataloader, 加快每次实验的数据读取速度.

4. See_graph.ipynb
如果保存了图, 可以用该程序生成相应热力图.

### 主函数 exp.py, 主要参数介绍
1. 记录类的参数
(可选) exp_ID,notes,exp_start_time: 为实验名称, 日志, 实验开始时间, 默认不填写. 若填写则将在程序的log输出功能 (以下介绍) 中保存这些信息.

2. save_log
(可选) 程序的日志输出类. 当该选项为True时, 将在 "args: root_path + log_path" 中保存本次实验的日志,该日志包含实验开始时间,ID,notes (若有),所有的超参数, 模型结构, 以及全部的输出结果 (即终端打印内容).
同时, 这些输出是默认加注日期的.

3. save_model
(可选) 保存每个 epoch 结果到 "args: root_path + saving_path" 中.

4. load_best
(可选) 每次epoch结束加载所保存的模型中, 具有最小vali loss的模型.

5. root_path
根存读目录. 模型所有保存或读取在此目录下进行.

6. data_path,adj_path,log_path,saving_path,exp_info_path
子目录. 分别为 数据集目录, 邻接矩阵的路径, 日志输出目录, 模型保存目录, 结果记录对象保存目录.

