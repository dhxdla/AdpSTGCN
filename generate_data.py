import argparse,torch,os
from utils.data_process import dataset_generator, raw_to_csv

parser = argparse.ArgumentParser()
parser.add_argument('--root_store_path',type = str,default = './',help = '')

'''raw data to csv'''
parser.add_argument('--raw_data_path',type = str,default = 'raw/',help = '')
parser.add_argument('--file_name',type = list,default = ['metr-la/metr-la.h5',
                                                         'pems-bay/pems-bay.h5',
                                                         'pems-04/pems04.npz',
                                                         'pems-08/pems08.npz'],help = '')

parser.add_argument('--csv_target_path',type = str,default = 'traffic/',help = '')
'''generate dtorch dataset'''
parser.add_argument('--csv_data_path',type = str,default = 'traffic/',help = '')
parser.add_argument('--dataset_target_path',type = str,default = 'traffic/',help = '')

parser.add_argument('--dataset_type',type = list,default = ['train','test','vali'],help = '')
parser.add_argument('--dtype',type = str,default = torch.float64, help = '')
parser.add_argument('--label_len',type = int,default = 12,help = '')
parser.add_argument('--pred_len',type = int,default = 12,help = '')
parser.add_argument('--scale',type = bool,default = True,help = '')
parser.add_argument('--freq',type = str,default = '15min',help = '')
parser.add_argument('--period_type',type = str,default = 'week',help = '')

args = parser.parse_args()

def generate_csv_file(args):
    name_list = args.file_name
    path_list = [(args.root_store_path + args.raw_data_path + e) for e in name_list]
    for i,raw_path in enumerate(path_list):
        file_path,file_type = name_list[i].split('.')
        file_name = file_path.split('/')[-1]
        target_path = args.root_store_path+args.csv_target_path
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        if file_type == 'h5':
            raw_to_csv.transform_file_h5(raw_path=raw_path,
                                         target_path=target_path,
                                         file_name=file_name)
        elif file_type == 'npz':
            if file_name == 'pems04':
                start_time = [2018,1,1,0,0]
            elif file_name == 'pems08':
                start_time = [2016,7,1,0,0]
            intervals = '5min'
            raw_to_csv.transform_file_npz(raw_path=raw_path,
                                          target_path=target_path,
                                          file_name=file_name,
                                          start_time=start_time,
                                          intervals=intervals)

def generate_torch_data_set(args):
    csv_file_path = args.root_store_path + args.csv_data_path
    name_list = os.listdir(csv_file_path)
    csv_name_list = []
    for name in name_list:
        if '.csv' in name:
            csv_name_list.append(name)
    target_path = args.root_store_path + args.dataset_target_path
    for file_name in csv_name_list:
        print('\t\"{}\" -> \"torch datasets\"'.format(file_name),end='')
        for flag in args.dataset_type:
            dataset = dataset_generator.Dataset_with_Time_Stamp(
                target_path,
                file_name,flag = flag,size = [args.label_len,args.pred_len],
                scale = args.scale,freq = args.freq,period_type = args.period_type,dtype=args.dtype)
            path = '{}{}{}/'.format(args.root_store_path,args.dataset_target_path,file_name[:-4])
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(dataset,'{}{}_dataset'.format(path,flag))
        print('\tdone.')

def main(args):
    # file path and names 
    print('\nraw to csv.')
    generate_csv_file(args)
    print('geneate torch dataset.')
    generate_torch_data_set(args)

if __name__  == '__main__':
    main(args)