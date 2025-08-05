#********************************************************************************
# vpr_run
#————————————————————————————————————————————————————————————————————————————————
# 功能描述：
#   0. 修改fpga arch file, 按照fpga_size, io_cap
#   1. 运行vpr结果保存在vpr_out对应的文件夹
#   2. 在class属性中保存各个过程文件的dir  
#————————————————————————————————————————————————————————————————————————————————


from xml.etree import ElementTree as ET
import os, sys, glob
from concurrent.futures import ProcessPoolExecutor


BENCHMARK_NAME = 'vtr7' #'extra_blif' #'mcnc_blif'
VPR_OUT_NAME = '../vpr_out_new' #'../vpr_out_extra' #'../vpr_out'
ARCH_NAME = 'EArch'

class VPR_RUN():
    def __init__(self, circuit_name, seed):

        #初始化变量
        self.circuit_name = str(circuit_name)
        self.seed = str(seed)
        self.circuit_full_name = self.circuit_name

        #创建vpr_out->circuit->circuit_seed 三级目录
        #初始化各文件夹地址
        self.script_dir = os.path.dirname(__file__)
        self.arch_blif_source_dir = os.path.abspath(os.path.join(self.script_dir,'../arch_blif_source'))
        #arch and bilf文件地址
        self.arch_dir = os.path.abspath(os.path.join(self.arch_blif_source_dir, ARCH_NAME+'.xml'))
        self.blif_dir = os.path.abspath(os.path.join(self.arch_blif_source_dir, BENCHMARK_NAME, self.circuit_full_name+'.blif')) 
        self.sdc_dir = os.path.abspath(os.path.join(self.arch_blif_source_dir, BENCHMARK_NAME, self.circuit_full_name+'.sdc')) 

        #创建vpr_out目录
        self.vpr_out_dir = os.path.abspath(os.path.join(self.script_dir, VPR_OUT_NAME))
        if not os.path.exists(self.vpr_out_dir): os.mkdir(self.vpr_out_dir)
        #创建vpr_circuit目录
        self.circuit_dir = os.path.abspath(os.path.join(self.vpr_out_dir, self.circuit_name))
        if not os.path.exists(self.circuit_dir): os.mkdir(self.circuit_dir)   

        #创建vpr_net_run目录
        self.circuit_pack_dir = os.path.abspath(os.path.join(self.circuit_dir, self.circuit_name+'_pack'))
        if not os.path.exists(self.circuit_pack_dir): os.mkdir(self.circuit_pack_dir)  
        self.net_dir = os.path.join(self.circuit_pack_dir, self.circuit_full_name+'.net')

        #创建当前circuit的seed输出目录
        self.circuit_seed_dir = os.path.abspath(os.path.join(self.circuit_dir, self.circuit_name+'_seed'+self.seed))
        if not os.path.exists(self.circuit_seed_dir): os.mkdir(self.circuit_seed_dir)
        self.place_dir = os.path.join(self.circuit_seed_dir, self.circuit_full_name+'.place')
        self.route_dir = os.path.join(self.circuit_seed_dir, self.circuit_full_name+'.route')
        self.vpr_stdout = os.path.join(self.circuit_seed_dir, 'vpr_stdout.log')


    #********************************************************************************
    # vpr_run
    #————————————————————————————————————————————————————————————————————————————————
    #fun功能：运行vpr and ace
    def run(self):
        if not os.path.exists(self.net_dir): 
            print(f"Error: The .net does not exist. Pls run pack_run first")
            sys.exit(1)
        # 在circuit_vpr_out_dir下执行对circuit的VPR，存放于circuit_vpr_out_dir
        circuit_vpr_run_name = self.circuit_name+'_seed'+ self.seed        
        if not os.path.exists(self.route_dir):
            print('My VPR_RUN log:', 'VPR Running by loading .net:', circuit_vpr_run_name)
            vpr_run_command = 'cd '+self.circuit_seed_dir+';'\
                +'$VTR_ROOT/vpr/vpr '+self.arch_dir+' '+self.blif_dir+\
                        ' --net_file '+self.net_dir+\
                        ' --place --route --analysis  --seed '+self.seed
            os.system(vpr_run_command)
        else:
            print('My VPR_RUN log:', 'VPR route exist:', circuit_vpr_run_name)

    def pack_run(self):
        if not os.path.exists(self.net_dir):
            print('My VPR_RUN log:', 'VPR Running .net:', self.circuit_name)
            vpr_run_command = 'cd '+self.circuit_pack_dir+';'\
                +'$VTR_ROOT/vpr/vpr '+self.arch_dir+' '+self.blif_dir+\
                    ' --pack '
            os.system(vpr_run_command)
        else:
            print('My VPR_RUN log:', 'VPR .net exist:', self.circuit_name)

if __name__ == '__main__':

    # 读取benchamrk内circuit name
    benchmark_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../arch_blif_source/vtr7'))
    blifs_path = glob.glob(os.path.join(benchmark_path, '*.blif'))
    benchmarks_list = [os.path.basename(path).replace('.blif', '') for path in blifs_path]


    def process_vpr(circuit_name, seed_i):
        vpr_run = VPR_RUN(circuit_name, seed=seed_i)
        vpr_run.pack_run() #先运行vtr packing
        vpr_run.run() #在运行vpr p&r


    # 创建一个包含多个进程的进程池
    with ProcessPoolExecutor(max_workers=10) as executor:
        # 为每个circuit_name和seed_i创建一个任务
        for circuit_name in benchmarks_list:  
            for seed_i in range(1):     
                executor.submit(process_vpr, circuit_name, seed_i)
