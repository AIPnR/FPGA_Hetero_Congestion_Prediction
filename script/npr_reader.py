from xml.etree import ElementTree as ET
import os, sys, re
import numpy as np
from vpr_run import VPR_RUN
from collections import defaultdict

#—————————————————————————————————————————
# npr_reader (net place route reader)
# 读取net place route文件并返回到对应的列表中
#—————————————————————————————————————————
class VPR_READER():
    def __init__(self, circuit_name, seed):

        vpr_run = VPR_RUN(circuit_name, seed)

        self.place_list =  self.place_reader(vpr_run.place_dir)
        # place_list：[ndoes_id, nodes_name, nodes_place(x,y), node_subblk]

        self.nets_id = self.net_id_reader(vpr_run.route_dir)
        # nets_id = {net_name: ID, }

        self.nodes_list = self.nodes_lister(vpr_run.net_dir, self.nets_id)
        # nodes_list:[ndoes_id，nodes_name, nodes_type, inputs_netname, outputs_netname]
    
        self.nets_list = self.nets_dictor(self.nodes_list)
        # nets_list(dict):['net_name': [nodes_id,...]]

        self.route_chan_width, self.fpga_size, self.critical_path, self.wl =\
            self.vpr_stdout_reader(vpr_run.vpr_stdout)

        self.chan_usage =  \
            self.route_conges_reader(vpr_run.route_dir, self.route_chan_width)
        # chanx_usage_*[x][y] = x,y 这个位置被用了几个track
        # chanx_usage_l4, chany_usage_l4, chanx_usage_l16, chany_usage_l16

        self.block_info_list, self.matrix = \
            self.architecture_reader(vpr_run.arch_dir, self.fpga_size)


    #********************************************************************************
    # nodes_lister
    # block_info_list, matrix = architecture_reader(self, arch_dir, fpga_size)
    #————————————————————————————————————————————————————————————————————————————————
    #fun功能：1. 把从arch提取出的block type信息存入block_info_list, 
    #        2. 按照fpga_size制作存有grid type的layout matrix
    #fun输入：arch_dir, fpga_size
    #fun输出：block_info_list(dict)：['type']: ['height': 1/4/8] ['one_hot': '1000000'] 
    #        matrix[x][y] = 'type'
    #———————————————————————————————————————————————————————————————
    def architecture_reader(self, arch_dir, fpga_size):
        print(os.path.basename(arch_dir), "reading...")
        root = ET.parse(arch_dir).getroot() # 获xml文件的内容取根标签
        pb_list = root.find("complexblocklist").findall('pb_type')
        block_info_list = {}
        for pb in pb_list:
            if pb.attrib.get('height'):
                block_info_list[pb.attrib.get('name')] = {'height': int(pb.attrib.get('height'))}
            else:
                block_info_list[pb.attrib.get('name')] = {'height': 1}
        block_info_list['EMPTY'] = {'height': 1}

        #增加type_id编码
        for i, key in enumerate(block_info_list.keys()):
            block_info_list[key]['type_id'] = i

        # 创建一个fpga_size_x x fpga_size_y的矩阵，初始值为'EMPTY', 并赋值W，H
        matrix = np.full((fpga_size), None)
        W, H = fpga_size
        # 获取所有的perimeter, col, row, single, fill元素，并按照priority排序
        layout = root.find("layout").find('fixed_layout')
        elements = layout.findall('./*[@priority]')
        elements.sort(key=lambda x: int(x.get('priority')), reverse=False)
        
        #创建layout矩阵
        for element in elements:
            element_type = element.get('type')
            if element.tag == 'perimeter':
                matrix[0, :] = element_type
                matrix[:, 0] = element_type
                matrix[-1, :] = element_type
                matrix[:, -1] = element_type

            elif element.tag == 'corners':
                matrix[0,0] = element_type
                matrix[-1, 0] = element_type
                matrix[-1, -1] = element_type
                matrix[0, -1] = element_type

            elif element.tag == 'single':         
                x = int(eval(element.get('x')))
                y = int(eval(element.get('y')))
                matrix[x, y] = element_type

            elif element.tag == 'fill':         
                matrix.fill(element_type)

            elif element.tag == 'col':         
                if set(element.keys()) == {'type', 'startx', 'starty', 'repeatx', 'priority'}:
                    startx = int(element.get('startx'))
                    starty = int(element.get('starty'))
                    repeatx = int(element.get('repeatx'))
                    height = int(block_info_list[element.get('type')]['height'])
                    if height>1:
                        block_info_list[element_type].update({
                            'startx': startx,
                            'starty': starty,
                            'repeatx': repeatx
                        })
                    end_y = H-starty
                    x_list = list(range(startx, W, repeatx))
                    while starty + height <= end_y:
                        matrix[x_list, starty:starty+height] = element_type
                        starty += height
                elif set(element.keys()) == {'type', 'startx', 'priority'}:
                    x = int(eval(element.get('startx')))
                    matrix[x, :] = element_type #只一列类似io
                else: raise ValueError("Unexpected keys in element col")

            elif element.tag == 'row':         
                if set(element.keys()) == {'type', 'starty', 'priority'}:
                    y = int(eval(element.get('starty')))
                    matrix[:, y] = element_type #只一行类似io
                else: raise ValueError("Unexpected keys in element row")
            else: raise ValueError("Unexpected type in element")
        
        return block_info_list, matrix



    #********************************************************************************
    # nodes_lister
    # nodes_list_temp = nodes_lister(self, net_dir, nets_id)
    #————————————————————————————————————————————————————————————————————————————————
    #fun功能：把从.net提取出的nodes信息存入nodes_list
    #fun输入：net_dir
    #fun输出：nodes_list：[ndoes_id, nodes_name, nodes_type, 
    #                    [inputs_netname, ... ], [outputs_netname, ...]]
    #———————————————————————————————————————————————————————————————
    def nodes_lister(self, net_dir, nets_id):
    #.net文件是pack flow输出的netlist文件
    #.net第一子级是CLB和IOPAD也就是place的nodes
    #.net子一级block形如<block name="n27" instance="clb[0]" mode="default">
    #提取其中nodes names 和 type(clb/io),
    #按顺序存入nodes_list行号即为nodes_id
        print(os.path.basename(net_dir), "reading...")
        root = ET.parse(net_dir).getroot() # 获xml文件的内容取根标签
        root_block = root.findall("block") #提取.net的block子一级
        nodes_list_temp = []
        nets_id_set = set(nets_id)  # 转换为集合以加快查找速度
        for nodes_elements in root_block:
            #for循环 遍历.net所有block子一级
            #提取nodes_name instance(type and id)
            nodes_name = nodes_elements.attrib.get('name')
            nodes_instance  = nodes_elements.attrib.get('instance')
            nodes_type, nodes_id = nodes_instance.split('[')
            nodes_id = nodes_id.rstrip(']')

            #提取inputs and outputs
            nodes_inputs = []
            for port in nodes_elements.find('inputs').findall('port'):
                nodes_inputs.extend(port.text.split(' '))
            nodes_inputs = set(nodes_inputs)
            #删除列表的中重复元素和open，gnd
            nodes_inputs.discard('open'); nodes_inputs.discard('gnd') 

            nodes_outputs = []
            for block in nodes_elements.findall('.//block'):
                if block.find('outputs'):
                    for port in block.find('outputs').findall('port'):
                        nodes_outputs.extend(port.text.split(' '))
            nodes_outputs = set(nodes_outputs).intersection(nets_id_set)
            #只提取存在于route的net

            nodes_list_temp.append([int(nodes_id), nodes_name, nodes_type, \
                                    nodes_inputs, nodes_outputs])
        return nodes_list_temp 


    #********************************************************************************
    # nets_dictor
    # nets_list = nets_dictor(self, nodes_list_temp)
    #————————————————————————————————————————————————————————————————————————————————
    #fun功能：把每条net连接到的nodes id存在字典中
    #        net是连接clb（次一级block）之间的线
    #fun输入：nodes_list
    #fun输出：nets_list(dict):['net_name': [nodes_names,...]]
    #———————————————————————————————————————————————————————   
    def nets_dictor(self, nodes_list_temp): 
        nets_list = defaultdict(list)
        for line_nodes_list in nodes_list_temp:
            # Merge inputs and outputs into a set to remove duplicates
            nodes_ioputs = set(line_nodes_list[3])
            nodes_ioputs.update(line_nodes_list[4])

            for net in nodes_ioputs:
                nets_list[net].extend([line_nodes_list[0]])
        return nets_list


    #********************************************************************************
    # net_id_reader
    # nets_id = net_id_reader(route_dir)
    #————————————————————————————————————————————————————————————————————————————————
    #fun功能: 读取route文件并以字典的形式输出net name对应的ID
    #fun输入: route_dir
    #fun输出: nets_id = {net_name: ID, }
    def net_id_reader(self, route_dir):
        # print(os.path.basename(route_dir), "reading...")

        with open(route_dir,'r') as route:
            route_line = route.read().splitlines() 
        nets_id = {}

        #读取route每一行，遇到Net关键字后，提取net所对应的ID
        for line in route_line:
            line_split=line.split()
            if len(line_split)>=3:
                # 读取net id
                if line_split[0]== 'Net':
                    net_name = line_split[2].split(')')[0].replace('(', '')
                    nets_id[net_name]= int(line_split[1])
        return nets_id 


    #********************************************************************************
    # place_reader
    # place_list = place_reader(place_dir)
    #————————————————————————————————————————————————————————————————————————————————
    #fun功能: 读取place文件并输出block坐标
    #fun输入: place_dir
    #fun输出: place_list：[ndoes_id, nodes_name, nodes_place(x,y), node_subblk]
    def place_reader(self, place_dir):
        print(os.path.basename(place_dir), "reading...")
        place_list=[]
        place_line=[]
        # print("\nplace loading and processing")
        #####读取转换了“\“的blif文件存放于 line_list  
        with open(place_dir,'r') as place:
            place_line = place.read().splitlines() 

        for i in range(5,len(place_line)):
            line=place_line[i].split()
            place_list.append([int(line[4].replace('#','')),line[0],[int(line[1]),int(line[2])], int(line[3])])
        # print(os.path.basename(place_dir), "save at place_list")
        # print('place_list:\n',place_list)
        return place_list


    #********************************************************************************
    # route_conges_reader
    # route_conges = route_conges_reader(route_dir)
    #————————————————————————————————————————————————————————————————————————————————
    #fun功能: 读取route文件并输出chanx, chany congestion ()
    #fun输入: route_dir
    #fun输出: route_conges[0]:chanx_usage, [1]:chany_usage]
    #        chanx_usage[x][y] = x,y这个位置被用了几个track
    def route_conges_reader(self, route_dir, channel_width):
        print(os.path.basename(route_dir), "reading...")

        with open(route_dir,'r') as route:
            route_line = route.read().splitlines() 

        fpga_size_x = int(route_line[1].split()[2])
        fpga_size_y = int(route_line[1].split()[4])

        chanx = np.zeros([channel_width, fpga_size_x, fpga_size_y])
        chany = np.zeros([channel_width, fpga_size_x, fpga_size_y])

        tracks_tmp = []
        #读取route每一行，把使用到的route和所在track用1在chanx和chany中标记
        for line in route_line:
            line_split=line.split()
            # print(line_split)
            if len(line_split)>=2:
                # CHANX所在行
                if line_split[2]== 'CHANX':
                    # print(line_split)
                    if len(line_split)==10:
                        x1, y1 = line_split[3].replace('(', '').replace(')', '').split(',')
                        x2, y2 = line_split[5].replace('(', '').replace(')', '').split(',')
                        track = line_split[7]
                        if y1 != y2: sys.exit() #route布线错误
                        for x in range(int(x1),int(x2)+1):
                            chanx[int(track)][x][int(y1)]=1

                        if abs(int(x1)-int(x2))==15:
                            tracks_tmp.append(int(track))

                    elif len(line_split)==8:
                        x, y = line_split[3].replace('(', '').replace(')', '').split(',')
                        track = line_split[5]
                        chanx[int(track)][int(x)][int(y)]=1
                    else: sys.exit() #route行描述未能识别

                # CHANY所在行
                if line_split[2]== 'CHANY':
                    # print(line_split)
                    if len(line_split)==10:
                        x1, y1 = line_split[3].replace('(', '').replace(')', '').split(',')
                        x2, y2 = line_split[5].replace('(', '').replace(')', '').split(',')
                        track = line_split[7]
                        if x1 != x2: sys.exit() #route布线错误
                        for y in range(int(y1),int(y2)+1):
                            chany[int(track)][int(x1)][y]=1
                    elif len(line_split)==8:
                        x, y = line_split[3].replace('(', '').replace(')', '').split(',')
                        track = line_split[5]
                        chany[int(track)][int(x)][int(y)]=1
                    else: sys.exit() #route行描述未能识别

        # strattix_arch, l4 track0-259, L16 260,299 channel_width = 300
        #求l4的数量，即l4 track id-1
        l4_track_num = round(channel_width * 260 / 300 / 2) * 2
        # assert min(tracks_tmp) >= l4_track_num, "wire segement id cal error"


        chanx_usage_l4 = np.zeros([fpga_size_x, fpga_size_y])
        chany_usage_l4 = np.zeros([fpga_size_x, fpga_size_y])
        for i in range(0, l4_track_num):
            chanx_usage_l4 += chanx[i]
            chany_usage_l4 += chany[i]

        chanx_usage_l16 = np.zeros([fpga_size_x, fpga_size_y])
        chany_usage_l16 = np.zeros([fpga_size_x, fpga_size_y])
        for i in range(l4_track_num, channel_width):
            chanx_usage_l16 += chanx[i]
            chany_usage_l16 += chany[i]

        return (chanx_usage_l4, chany_usage_l4, chanx_usage_l16, chany_usage_l16)


    def vpr_stdout_reader(self, vpr_stdout_dir):
        print(os.path.basename(vpr_stdout_dir), "reading...")
        with open(vpr_stdout_dir, 'r') as file:
            log_data = file.read()

        route_chan_width = None
        match_width = re.search(r'Best routing used a channel width factor of (\d+)', log_data)
        if match_width:
            route_chan_width = int(match_width.group(1))
        
        match_size = re.search(r'## Build Device Grid\nFPGA sized to (\d+) x (\d+)', log_data)
        if match_size:
            size_x, size_y = map(int, match_size.groups())

        match_critical_path = re.search(r"Final critical path: (\d+\.\d+)", log_data) 
        if match_critical_path: 
            critical_path = float(match_critical_path.group(1))

        match_wirelength = re.search(r"Total wirelength: (\d+)", log_data) 
        if match_wirelength: 
            wl = int(match_wirelength.group(1))
        
        return route_chan_width, (size_x, size_y), critical_path, wl



# ********************************************************************************
# banp_read test
# ————————————————————————————————————————————————————————————————————————————————
if __name__ == '__main__':

    vpr_reader = VPR_READER('minres', seed=0)

    # print('read place_list:  block num =  ', len(vpr_reader.place_list))
    # print('read nets_id:     net num =    ', len(vpr_reader.nets_id))
    # print('read nodes_lis:   block num =  ', len(vpr_reader.nodes_list))
    # print('read nets_list:   net num =    ', len(vpr_reader.nets_list))
    # print('read vpr_std.log: chan_width=  ', vpr_reader.route_chan_width)
    # print('read vpr_std.log: fpga size :  ', vpr_reader.fpga_size)
    # print('read arch      :  block type_n:', len(vpr_reader.block_info_list))
    # print('read arch      :  fpga size :  ', vpr_reader.matrix.shape)

    # chanx_usage_l4, chany_usage_l4, chanx_usage_l16, chany_usage_l16 =  \
    #      vpr_reader.chan_usage
    # print('read chan_usage:  x_l4 shape:  ', chanx_usage_l4.shape)

    print(vpr_reader.route_chan_width, vpr_reader.fpga_size, vpr_reader.critical_path, vpr_reader.wl)

    pass
    



