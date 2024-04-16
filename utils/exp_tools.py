import os,torch,time,copy
import numpy as np
from utils.loadAsave_tools import load_adj

def my_bool(s):
    return s !=  'False'

def setting_seed(seed = 1):
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 为GPU设置随机种子
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = False

def adjust_learning_rate(optimizer, epoch, adj_lr, latest_test_loss):
    log = None
    if not adj_lr:
        return None
    if not epoch < 2:
        if np.mean(latest_test_loss[:-1]) <= latest_test_loss[-1]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/2
            log = '\n\tUpdating learning rate to <{}>\n'.format(param_group['lr'])
    return log

class log_writer():
    def __init__(self,args,model_info,
                 print_while_wirte = True,
                 output_with_time = True,
                 max_len_every_line = 100): 
        
        self.args = args
        self.print_while_wirte = print_while_wirte
        self.output_with_time = output_with_time

        self.max_len_every_line = max_len_every_line
        self.line = '-'*(self.max_len_every_line+21) if (output_with_time) else '-'*self.max_len_every_line
        
        self.creat_log_file()
        self.write_title()
        self.write_notes()
        self.write_args_info()
        self.write_model_structure(model_info)
    
    def start_epoch(self,epoch,flag,report=True):
        if flag == 'exp':
            log = '\nEpoch: {:03d}'.format(epoch)
            self.output_log(log)
        elif report:
            log = 'State: <{0:s}>\n\t{1:8}{2:<12}{3:<12}{4:<12}{5:<12}{6:<16}{7:<12}'.format(flag,'iter','Loss','MAE','MAPE','RMSE','Speed (/iter)','Time Cost')
            self.output_log(log,start = '\t')

    def end_epoch(self,summary_log,lr_log,load_log):
        if not summary_log == '':
            # title
            log = summary_log['title']
            self.output_log(log)
            log = '\t{0:8}{1:12}{2:12}{3:12}{4:12}{5:12}'.format('State','Avg. Loss','Avg. MAE','Avg. MAPE','Avg. RMSE','Time Cost')
            self.output_log(log)
            # body 1
            for flag in ['train','vali','test']:
                self.output_log(summary_log[flag], start='\t')
            # body 2
            self.output_log(summary_log['last5title'], start='\t')
            self.output_log(summary_log['last5'], start='\t')
        if not lr_log == None:
            self.output_log(lr_log, start='\t')
        if not load_log == None:
            self.output_log(load_log, start='\t')
        self.write_line(line_type='long')

    def end_exp(self):
        log = "Training finished"
        self.output_log(log,start='\t')

    def write_iterinfo(self,log):
        self.output_log(log,start = '\t')

    def write_title(self):
        self.write_line('long')
        log = 'Experiment Start: <{}>'.format(self.args.exp_start_time)
        self.output_log(log,print_while_wirte = True)
        self.write_line('long')

    def write_notes(self):
        if self.args.notes  == '':
            return
        self.output_log('Notes:\n',print_while_wirte = False)
        self.output_log(self.args.notes,print_while_wirte = False,start = '\t')
        self.write_line('long')

    def write_args_info(self):
        args_list = self.args._get_kwargs()
        exp_info = self.get_str_exp_info(args_list)
        for item in exp_info:
            print(item,end = '')
        self.output_log(exp_info,print_while_wirte = False)
        self.write_line('long')

    def write_model_structure(self,model_info):
        self.output_log('Model structure:',print_while_wirte = False)
        self.output_log(str(model_info),print_while_wirte = False,start = '\t')
        self.write_line('long')

    def creat_log_file(self):
        if not self.args.save_log: # 不写log时跳过
            return
        if self.args.root_path  == '':
            # 绝对路径
            abs_path = os.getcwd()
            folder_path = '{}/{}{}'.format(abs_path,self.args.root_path,self.args.log_path)
        else:
            folder_path = self.args.root_path + self.args.log_path
        # 创建目录
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # 创建文件
        file_name = self.args.exp_start_time + '.txt'
        self.file_path = folder_path + file_name
        note = open(self.file_path,'w')
        note.close()

    def output_log(self,log,print_while_wirte = None,start = '',end = '\n'):
        if print_while_wirte  == None: # 没有指定时,使用self的
            print_while_wirte = self.print_while_wirte
        if print_while_wirte: # 打印信息
            print(log,end = end)
        if not self.args.save_log: # 不写log时跳过
            return
        file = open(self.file_path,'a')
        if type(log)  == str:# 处理为list
            log = [log]
        if type(log)  == list:
            for item in log:
                tmp_str = item.split('\n')
                for text in tmp_str[:-1]:
                    self.self_write(file,text+'\n',start,end)
                if not tmp_str[-1]  == '':
                    self.self_write(file,tmp_str[-1],start,end)
        file.close()
        
    def self_write(self,file,text,start = '',end = ''):
        if not start  == None:
            text = start + text
        if text[-1]  == '\n' and end  == '\n':
            pass
        else:
            text = text + end
        if self.output_with_time:
            now_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            if not self.line in text:
                text = '{}| {}'.format(now_time,text) if not text  == '\n' else ' '*19 + '|'
            file.write(text)
        else:
            file.write(text)

    def get_str_exp_info(self,args_list):
        len_line = 0
        string_list = []
        string_list.append('Exp Setting:\n')
        # string_list.append('{}\n'.format(self.line))
        string = ''
        interval_string = ': \"'
        end_string = '\";  '
        for item in args_list:
            flag = 0
            # exp info
            key = item[0]
            value = item[1]
            str_value = str(value)
            # 将写入的log
            item_log = ' '*4 + key + interval_string + str_value + end_string
            len_line += len(item_log)
            # 判断是否超出最大长度
            if len_line <= self.max_len_every_line:
                string += item_log
            else:
                # w
                string += '\n'
                string_list.append(string)
                # 重置
                string = '' + item_log
                len_line = len(item_log)
                flag = 1 
        if (not len(string) == 0) and (flag == 0):
            string_list.append(string+'\n')
        # string_list.append(string)
        return string_list

    def write_line(self,line_type = 'short',end = '\n'):
        if not self.args.save_log:
            return
        note = open(self.file_path,'a')
        if line_type  == 'short':
            note.write(' '*19+'|'+self.line[20:]+end)
        elif line_type  == 'long':
            note.write(self.line+end)
        note.close()

    def write_blank(self,blank_type = 'short',end = '\n'):
        if not self.args.save_log:
            return
        note = open(self.file_path,'a')
        if blank_type  == 'short':
            note.write(' '*19+'|'+end)
        elif blank_type  == 'long':
            note.write(' '+end)
        note.close()

class exp_infomation():
    def __init__(self):

        self.metrics_info = {
            'train':{},
            'vali':{},
            'test':{},
            'hyperparameters':{}
            }
        
        self.using_time = {
            'train':[],
            'vali':[],
            'test':[]}
        
    def update(self,epoch,metrics,using_time = None,flag = None):
        # append using time
        if not using_time  == None:
            self.using_time[flag].append(using_time)
        # append metrics
        for idx in metrics.keys():
            # check keys of dict
            if not (idx in self.metrics_info[flag].keys()):
                self.metrics_info[flag].update({idx:[]})
            # append new item
            self.metrics_info[flag][idx].append(metrics[idx])

    def get_best(self):
        self.best_metrics  = {
            'train':{},
            'vali':{},
            'test':{}}
        for state in list(self.best_metrics.keys()):
            for idx in list(self.metrics_info[state].keys()):
                if idx[0] =='m':
                    self.best_metrics[state].update({('best'+idx[1:]):np.min(self.metrics_info[state][idx])})
        return self.best_metrics
    
    def get_metrics(self):
        return self.metrics_info
    
    def log_setting(self,configs):
        args_dict = {
            'seq_len':configs['seq_len'],
            'd_model':configs['d_model'],
            'kernel_size':configs['kernel_size'],
            'data_path':configs['data_path']}
        self.update(0, args_dict, flag='hyperparameters')

class metrics_info():
    def __init__(self):
        self.init_metrics()
    
    def init_metrics(self):
        self.start_time = time.time()
        self.last_time = time.time()
        self.metrics = {'loss':[], 'mae':[],'mape':[], 'rmse':[], 'mae_all':[], 'mape_all':[], 'rmse_all':[],
                        'm_loss':np.inf, 'm_mae':np.inf, 'm_mape':np.inf, 'm_rmse':np.inf,
                        'using_time':None}

    def update(self,loss,metrics):
        self.metrics['loss'].append(loss)
        self.metrics['mae'].append(metrics[0])
        self.metrics['mape'].append(metrics[1])
        self.metrics['rmse'].append(metrics[2])
        self.metrics['mae_all'].append(metrics[3])
        self.metrics['mape_all'].append(metrics[4])
        self.metrics['rmse_all'].append(metrics[5])

    def update_avg(self):
        self.metrics['m_loss'] = (np.mean(self.metrics['loss']))
        self.metrics['m_mae'] = (np.mean(self.metrics['mae']))
        self.metrics['m_mape'] = (np.mean(self.metrics['mape']))
        self.metrics['m_rmse'] = (np.mean(self.metrics['rmse']))
        self.metrics['using_time'] = time.time() - self.start_time

    def get_iterinfo(self,iters,print_every):
        num_iter = 1 if iters==0 else print_every
        before = 0 if (iters==0) else (iters-print_every)
        later = 1 if (iters==0) else iters
        log = '\t{0:04d}{1:4s}{2:<12.4f}{3:<12.4f}{4:<12.4%}{5:<12.4f}{6:<16s}{7:<12s}'
        log = log.format(iters,' ',
                np.mean(self.metrics['loss'][before:later]),
                np.mean(self.metrics['mae'][before:later]),
                np.mean(self.metrics['mape'][before:later]),
                np.mean(self.metrics['rmse'][before:later]),
                '{:.4f} s'.format((time.time()-self.last_time)/num_iter),
                '{:.2f} s'.format(time.time()-self.start_time))
        self.last_time = time.time()
        return log
    
    def get_metrics(self):
        return self.metrics 

def exp_summary(epoch, metrics):
    log = 'Info Report: <Epoch {:03d}> '.format(epoch)
    summary_log = {'title':log, 'train':None, 'vali':None, 'test':None, 'last5':None, 'last5title':None}
    for flag in ['train','vali','test']:
        log = '\t{0:5s}{1:3s}{2:<12.4f}{3:<12.4f}{4:<12.4%}{5:<12.4f}{6:<12s}'
        log = log.format(flag,'',
            metrics[flag]['m_loss'][epoch-1],          
            metrics[flag]['m_mae'][epoch-1],
            metrics[flag]['m_mape'][epoch-1],
            metrics[flag]['m_rmse'][epoch-1],
            '{:.2f} s'.format(metrics[flag]['using_time'][epoch-1]))
        summary_log[flag] = log
    # latest 5 loss
    latest_test_loss = copy.deepcopy(metrics['test']['m_mae'])[-5:]
    title_log = '<Latest 5>\t'
    log = '{:4s}{}\t'.format(' ','Loss')
    for idx in range(len(latest_test_loss)):
        title_log +=  '{:<8d}\t'.format(1+idx+epoch-len(latest_test_loss))
        log += '{:<8.4f}\t'.format(latest_test_loss[idx])
    summary_log['last5title'] = title_log
    summary_log['last5'] = log
    return latest_test_loss,summary_log