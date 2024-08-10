# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# Author: FANG Junying, fangjunying@neuracle.cn

# Versions:
# 	v0.1: 2018-08-14, orignal

# Copyright (c) 2016 Neuracle, Inc. All Rights Reserved. http://neuracle.cn/

from neuracle_lib.dataServer import (dataserver_thread)
import time
from time import sleep
import random
if __name__ == '__main__':
    N = 0
    flagstop = False

    neuracle = dict(device_name='Neuracle', hostname='127.0.0.1', port=8712,
                    srate=1000, chanlocs=['Pz', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'Oz', 'O1', 'O2', 'TRG'], n_chan=64)

    ### pay attention to the device you used
    target_device = neuracle
    srate = target_device['srate']
    print('!!!! The type of device you used is %s' % target_device['device_name'])
    ## init dataserver
    time_buffer = 1  # second
    thread_data_server = dataserver_thread(threadName='data_server', device=target_device['device_name'],
                                           n_chan=target_device['n_chan'],
                                           hostname=target_device['hostname'], port=target_device['port'],
                                           srate=target_device['srate'], t_buffer=time_buffer)
    thread_data_server.Daemon = True
    notconnect = thread_data_server.connect()
    if notconnect:
        raise TypeError("Can't connect recorder, Please open the hostport ")
    else:
        thread_data_server.start()
        print('Data server connected')


    # def compare_data_to_range(data, lower_bound, upper_bound):
    #     """将数据与设定范围比较并输出相应标签"""
    #     data_mean = sum(data) / len(data) if data else 0
    #     if data_mean > upper_bound:
    #         return "左位置抓取"  # 高于范围
    #     elif lower_bound <= data_mean <= upper_bound:
    #         return "中位置抓取"  # 在范围内
    #     else:
    #         return "右位置抓取"  # 低于范围

    try:
        while not flagstop:  # get data in one second step
            nUpdate = thread_data_server.get_bufferNupdate()
            if nUpdate > (1 * srate - 1):
                N += 1
                data = thread_data_server.get_bufferData()
                thread_data_server.set_bufferNupdate(0)

                # current_time = time.time()
                # if current_time - last_check_time >= 30:  # 每30秒检查一次
                #     # 定义范围
                #     lower_bound = 10  # 设置为实际范围
                #     upper_bound = 50  # 设置为实际范围
                #
                #     # 调用比较函数并输出标签
                #     label = compare_data_to_range(data, lower_bound, upper_bound)
                #     print(f"Label: {label}")
                #     last_check_time = current_time

                print(data)
                # print(data.shape)
                time.sleep(0.99)

            if N > 30:
                flagstop = True
    except:
        pass

    thread_data_server.stop()





def compare_data_to_range():
            # sleep(30)
            random.seed(time.time())
            labels = ["左位置抓取", "中位置抓取", "右位置抓取"]
            generated_label = random.choice(labels)
                        # print(generated_label)
            return generated_label


