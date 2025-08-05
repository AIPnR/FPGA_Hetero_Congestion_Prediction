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
VPR_OUT_NAME = '../vpr_out' #'../vpr_out_extra' #'../vpr_out'
VPR_OUT_NAME_WCPNET = '../vpr_out_wcpnet' #'../vpr_out_extra' #'../vpr_out'
ARCH_NAME = 'EArch_new'

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

        #创建vpr_out_wcpnet目录
        self.vpr_out_wcpnet_dir = os.path.abspath(os.path.join(self.script_dir, VPR_OUT_NAME_WCPNET))
        if not os.path.exists(self.vpr_out_wcpnet_dir): os.mkdir(self.vpr_out_wcpnet_dir)
        #创建vpr_circuit_wcpnet目录
        self.circuit_wcpnet_dir = os.path.abspath(os.path.join(self.vpr_out_wcpnet_dir, self.circuit_name))
        if not os.path.exists(self.circuit_wcpnet_dir): os.mkdir(self.circuit_wcpnet_dir) 
        self.circuit_seed_wcpnet_dir = os.path.abspath(os.path.join(self.circuit_wcpnet_dir, self.circuit_name+'_seed'+self.seed))
        if not os.path.exists(self.circuit_seed_wcpnet_dir): os.mkdir(self.circuit_seed_wcpnet_dir) 


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
    # def wcpmet_img(self):
    #     vpr_run_command = 'cd '+self.circuit_seed_wcpnet_dir+';'\
    #         +'$VTR_NEW_ROOT/vpr/vpr '+self.arch_dir+' '+self.blif_dir+\
    #                 ' --net_file ' +self.net_dir+\
    #                 ' --place_file ' +self.place_dir+\
    #                 ' --seed ' +self.seed+\
    #                 ' --router_lookahead classic '+\
    #                 ' --graphics_commands '+\
    #                 '\"set_draw_block_text 0; set_draw_block_outlines 0; save_graphics place.png; '+\
    #                 'set_nets 1; save_graphics net.png; set_nets 0; ' +\
    #                 'set_set_draw_block_pin_util 1; save_graphics pin_util.png; exit 0\" '
    #     print(vpr_run_command)
    #     os.system(vpr_run_command)

    def wcpmet_img(self):
        vpr_run_command = 'cd '+self.circuit_seed_wcpnet_dir+';'\
            +'$VTR_NEW_ROOT/vpr/vpr '+self.arch_dir+' '+self.blif_dir+\
                    ' --seed ' +self.seed+\
                    ' --router_lookahead classic '+\
                    ' --graphics_commands '+\
                    '\"set_draw_block_text 0; set_draw_block_outlines 0; save_graphics place.png; '+\
                    'set_nets 1; save_graphics net.png; set_nets 0; ' +\
                    'set_set_draw_block_pin_util 1; save_graphics pin_util.png; exit 0\" '
        print(vpr_run_command)
        os.system(vpr_run_command)



if __name__ == '__main__':

    # 读取benchamrk内circuit name
    benchmark_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../arch_blif_source/vtr7'))
    blifs_path = glob.glob(os.path.join(benchmark_path, '*.blif'))
    benchmarks_list = [os.path.basename(path).replace('.blif', '') for path in blifs_path]

    # vpr_run = VPR_RUN('mcml', 0)
    # vpr_run.wcpmet_img() #在运行vpr p&r

    def process_vpr(circuit_name, seed_i):
        vpr_run = VPR_RUN(circuit_name, seed_i)
        vpr_run.wcpmet_img() #在运行vpr p&r

    # 创建一个包含多个进程的进程池
    with ProcessPoolExecutor(max_workers=1) as executor:
        # 为每个circuit_name和seed_i创建一个任务
        for circuit_name in benchmarks_list:  
            for seed_i in range(20,21):     
                executor.submit(process_vpr, circuit_name, seed_i)
