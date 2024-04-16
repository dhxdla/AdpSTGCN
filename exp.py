import argparse,time,torch
from utils import loadAsave_tools
from experiment.engine import *
from utils.exp_tools import my_bool,setting_seed
parser = argparse.ArgumentParser()

'''exp environment & setting'''
# Info
parser.add_argument('--exp_ID', type = str, default = '', help = '实验ID')
parser.add_argument('--notes', type = str, default = '', help = '记录实验信息')
parser.add_argument('--exp_start_time', type = str, default = '', help = '实验开始时间,自动填写.')
parser.add_argument('--save_log', type = my_bool, default = True, help = '是否输出日志文件')
parser.add_argument('--print_every', type = int, default = 100,  help = 'print training information every x iters')
# env
parser.add_argument('--device', type = str, default = 'cuda:7', help = '设备')
parser.add_argument('--dtype', type = str, default = torch.float, help = '数据类型')
parser.add_argument('--adj_lr', type = my_bool, default = False, help = '调节Learning rate')
parser.add_argument('--load_best', type = my_bool, default = False, help = '每个epoch加载先前最好vali模型')
parser.add_argument('--save_model', type = my_bool, default = False, help = '每个epoch保存模型')
# root path
parser.add_argument('--root_path', type = str, default = './',help = '存写主目录') # /users/pp/data_base/Traffic_prediction/
                                                                                                    # /data1/pp21/TP_store/
# data path
parser.add_argument('--data', type=str)                                                                                         # /data/pp21/TP_2_store_folder/
parser.add_argument('--data_path', type = str, default = 'traffic/pems-bay/',help = '数据集目录')      # metr-la, pems-bay,pems04/08,_flow,_occ,_speed
parser.add_argument('--adj_path', type = str, default = 'raw/pems-bay/pems_bay_adj.pkl',help = '邻接矩阵路径') # metr-la/metr_la_adj.pkl,pems-bay/pems_bay_adj.pkl,pems-04/distance.csv
# result saving
parser.add_argument('--log_path', type = str, default = 'results/logs/',help = '日志输出目录')
parser.add_argument('--saving_path', type = str, default = 'results/loss/',help = '模型保存目录')
parser.add_argument('--exp_info_path', type = str, default = 'results/dict/',help = '实验记录类保存目录')
# Data Info
parser.add_argument('--n_nodes', type = int, default = 325, help = '节点数') # metr-la:207, pems-bay:325, pems04:307, pems08:170
parser.add_argument('--seq_len', type = int, default = 12, help = '输入长度')
parser.add_argument('--predict_len', type = int, default = 12, help = '预测长度')
parser.add_argument('--dim_date', type = int, default = 5, help = '日期数据维度')
# Optimal
parser.add_argument('--epochs', type = int, default = 100, help = 'epochs')
parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size')
parser.add_argument('--optimizer', type = str, default = 'adam', help = '优化器')
parser.add_argument('--learning_rate', type = float, default = 0.002, help = '学习率')
parser.add_argument('--weight_decay', type = float, default = 1e-4, help = 'weight decay')
parser.add_argument('--dropout', type = float,  default = 0.3, help = 'drop out')

'''exp hyper parameters'''
# graph info
parser.add_argument('--num_topo_adj', type = int, default = 2, help = '拓扑图数量')
# hyper params
parser.add_argument('--gamma_lower', type = float, default = -0.1, help = '超参数gamma low')
parser.add_argument('--gamma_upper', type = float, default = 1.1, help = '超参数gamma up')
parser.add_argument('--addaptadj', type = my_bool, default = False, help = 'Graph wavenet 自适应矩阵')
# Model Structure
parser.add_argument('--kernel_size', type = int, default = 2, help = '时间卷积核size') # 2,2,2
parser.add_argument('--n_layers', type = int, default = 2, help = '模型层数')
parser.add_argument('--n_blocks', type = int, default = 4, help = '层block数')
# FC dim
parser.add_argument('--in_dim', type = int, default = 1, help = '')
parser.add_argument('--out_dim', type = int, default = 12, help = '')
parser.add_argument('--d_model', type = int, default = 32, help = '隐藏表示维度')
parser.add_argument('--skip_channels', type = int, default = 256, help = '')
parser.add_argument('--end_channels',type = int, default = 512, help = '')

'''compile'''
args = parser.parse_args()

if args.data == 'METRLA':
    args.data_path = 'traffic/metr-la/'
    args.adj_path = 'raw/metr-la/metr_la_adj.pkl'
    args.n_nodes = 207

if args.data == 'BAY':
    args.data_path = 'traffic/pems-bay/'
    args.adj_path = 'raw/pems-bay/pems_bay_adj.pkl'
    args.n_nodes = 325

# check cuda, 如果 gpu不可用则调整为cpu
if not torch.cuda.is_available():
    print('Cuda not available. device -> cpu')
    args.device = torch.device('cpu')
# 实验开始时间
args.exp_start_time = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time())) if (args.exp_start_time=='') else args.exp_start_time
# check load best settings, load best必须以save model为前提.
if args.load_best:
    args.save_model =  args.load_best

# main function
def main(args):
    setting_seed(42) # 42: all the truths of the universe
    dataset,dataloader,scaler = loadAsave_tools.load_data(args)
    scaler.to_tensor(args.device)
    exp = Engine(args, dataset, dataloader, scaler)
    exp.run_exp()

# main function
if __name__ == '__main__':
    main(args)