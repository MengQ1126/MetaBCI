import math

from psychopy import monitors
import numpy as np
from metabci.brainstim.paradigm import (
    SSVEP,
    P300,
    MI,
    AVEP,
    SSAVEP,
    paradigm,
    pix2height,
    code_sequence_generate,
)
from metabci.brainstim.framework import Experiment
from psychopy.tools.monitorunittools import deg2pix

if __name__ == "__main__":
    mon = monitors.Monitor(
        name="primary_monitor",
        width=59.6,
        distance=60,  # width 显示器尺寸cm; distance 受试者与显示器间的距离
        verbose=False,
    )
    mon.setSizePix([1920, 1080])  # 显示器的分辨率
    # mon.setSizePix([960, 640])  # 显示器的分辨率
    mon.save()
    bg_color_warm = np.array([0, 0, 0])
    # win_size = np.array([960, 640])
    win_size = np.array([1920, 1080])
    # esc/q退出开始选择界面
    ex = Experiment(
        monitor=mon,
        bg_color_warm=bg_color_warm,  # 范式选择界面背景颜色[-1~1,-1~1,-1~1]
        screen_id=0,
        win_size=win_size,  # 范式边框大小(像素表示)，默认[1920,1080]
        is_fullscr=False,  # True全窗口,此时win_size参数默认屏幕分辨率
        record_frames=False,
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    )
    win = ex.get_window()

    # q退出范式界面
    """
    SSVEP
    """
    n_elements, rows, columns = 3, 1, 3  # n_elements 指令数量;  rows 行;  columns 列
    # n_elements, rows, columns = 4, 2, 2  # n_elements 指令数量;  rows 行;  columns 列
    stim_length, stim_width = 500, 500  # ssvep单指令的尺寸
    stim_color, tex_color = [1, 1, 1], [1, 1, 1]  # 指令的颜色，文字的颜色
    fps = 240  # 屏幕刷新率
    stim_time = 2  # 刺激时长
    stim_opacities = 1  # 刺激对比度
    freqs = np.arange(8, 16, 0.4)  # 指令的频率
    phases = np.array([i * 0.35 % 2 for i in range(n_elements)])  # 指令的相位

    basic_ssvep = SSVEP(win=win)

    basic_ssvep.config_pos(
        n_elements=3,
        rows=rows,
        columns=columns,
        stim_length=stim_length,
        stim_width=stim_width,
    )
    basic_ssvep.config_text(tex_color=tex_color)
    basic_ssvep.config_color(
        refresh_rate=fps,
        stim_time=stim_time,
        stimtype="sinusoid",
        stim_color=stim_color,
        stim_opacities=stim_opacities,
        freqs=freqs,
        phases=phases,
    )
    basic_ssvep.config_index()
    basic_ssvep.config_response()

    bg_color = np.array([0.3, 0.3, 0.3])  # 背景颜色
    display_time = 1  # 范式开始1s的warm时长
    index_time = 1  # 提示时长，转移视线
    rest_time = 0.5  # 提示后的休息时长
    response_time = 1  # 在线反馈
    port_addr = "COM8"  #  0xdefc                                  # 采集主机端口
    port_addr = None  #  0xdefc
    nrep = 2  # block数目
    lsl_source_id = "meta_online_worker"  # None                 # source id
    online = False  # True                                       # 在线实验的标志
    ex.register_paradigm(
        "basic SSVEP",
        paradigm,
        VSObject=basic_ssvep,
        bg_color=bg_color,
        display_time=display_time,
        index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,
        nrep=nrep,
        pdim="ssvep",
        lsl_source_id=lsl_source_id,
        online=online,
    )

    ex.run()

from scipy import signal
from scipy.signal import hilbert
import cmath
from neuracle_lib.dataServer import DataServerThread

import threading
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType
from time import sleep
import numpy as np
import pickle
import time
import warnings
warnings.filterwarnings("ignore")


def connect_robot():
    try:
        ip = "192.168.5.1"
        dashboard_p = 29999
        move_p = 30003
        feed_p = 30004
        print("正在建立连接...")
        dashboard = DobotApiDashboard(ip, dashboard_p)
        move = DobotApiMove(ip, move_p)
        feed = DobotApi(ip, feed_p)
        print(">.<连接成功>!<")
        return dashboard, move, feed
    except Exception as e:
        print(":(连接失败:(")
        raise e

def run_point(move: DobotApiMove, point_list: list):
    move.JointMovJ(point_list[0], point_list[1], point_list[2], point_list[3], point_list[4], point_list[5])

def get_feed(feed: DobotApi):
    global current_actual
    hasRead = 0
    while True:
        data = bytes()
        while hasRead < 1440:
            temp = feed.socket_dobot.recv(1440 - hasRead)
            if len(temp) > 0:
                hasRead += len(temp)
                data += temp
        hasRead = 0

        a = np.frombuffer(data, dtype=MyType)
        if hex((a['test_value'][0])) == '0x123456789abcdef':

            # Refresh Properties
            current_actual = a["tool_vector_actual"][0]
            # print("tool_vector_actual:", current_actual)
        sleep(0.001)

def wait_arrive(point_list):
    global current_actual
    while True:
        is_arrive = True

        if current_actual is not None:
            for index in range(len(current_actual)):
                if (abs(current_actual[index] - point_list[index]) > 1):
                    is_arrive = False

            if is_arrive:
                return

        sleep(0.001)

# def PhaseSI(xr, yr):
#     x = hilbert(xr)
#     y = hilbert(yr)
#     x_theta = np.angle(x)
#     y_theta = np.angle(y)
#     xy_theta = x_theta - y_theta
#     aa = [cmath.exp(x) for x in (0.0000 + 1.0000j) * xy_theta]
#     PLV = abs(sum(aa)) / len(xr)
#     return PLV
#
# def data_psi(data):
#     [M, N] = data.shape
#     if M> N:
#         data = data.T
#         [M, N] = data.shape
#     PLV = np.zeros((M, M))  # 分段的尺度
#     for i in range(M-1):
#         for j in range(1, M):
#             PLV[i, j] = PhaseSI(data[i, :], data[j, :])
#             PLV[j, i] = PLV[i, j]
#     PLV = PLV + np.identity(M)
#     return PLV
#
# def getcsp(eegdata):
#     downsampled = signal.resample(eegdata, int(len(eegdata[1]) / 10), axis=1)
#     print('降采样:', downsampled.shape)
#
#     # 滤波
#     Hz, min_filt, max_filt = 1000 / 10, 8, 30
#     min_stop, max_stop = 2 * min_filt / Hz, 2 * max_filt / Hz
#     butter_b, butter_a = signal.butter(8, [min_stop, max_stop], 'bandpass')  # 配置滤波器 8 表示滤波器的阶数
#     filtedData = signal.filtfilt(butter_b, butter_a, downsampled, axis=1)  # data为要过滤的信号
#     return filtedData
#
# def Connect_PLV(inte_data):
#     A = data_psi(inte_data)
#     return A
#
# def checklabel(a,b):
#     cc = 0
#     c = 0
#     if a > 0:
#         for i in range(len(b)):
#             if int(b[i]) == 1065353216:
#                 print('-------------------------------------------------一号动作')
#                 cc = 1
#                 c += i
#             elif int(b[i]) == 1073741824:
#                 print('--------------------------------------------------二号动作')
#                 cc = 2
#                 c += i
#             elif int(b[i]) == 1077936128:
#                 print('------------------------------------------------三号动作')
#                 cc = 3
#                 c += i
#             elif b[i] == 0:
#                 c += 0
#
#             else:
#                 pass
#     else:
#         c = 0
#
#     return c, cc
#
# def getindex(N, whereihex):
#     if whereihex != 0:
#         index = N*3000-3000+whereihex
#     else:
#         index = 0
#     return int(index)
#

def main():
    # 配置设备
    dashboard, move, feed = connect_robot()
    print("开始上电...")
    dashboard.PowerOn()
    y = int(input("已上电，使能请按1，否则退出:  "))
    if y == 1:
        print("开始使能...")
        dashboard.EnableRobot()
        print("完成使能:)")
        feed_thread = threading.Thread(target=get_feed, args=(feed,))
        feed_thread.setDaemon(True)
        feed_thread.start()
        # dashboard.ToolDOExecute(1,0)


        # point_a1 = [4.78,0.64,128,-39,-90,50]  # 左抓
        # point_a2 = [4.5,-8,114.4,-16,-89,49.4]  # 左抬
        # point_a3 = [-10.4,24,89.5,-23,-89.5,34.5]  # 左放
        # point_b1 = [-32,-4.2,135,-45,-90,14]  # 右抓
        # point_b2 = [-32,-12,126,-27,-90,14]  # 右抬
        # point_b3 = [-20,27,80,-15.8,-90,24.7]  # 右放
        # point_c = [-4,-28,114,2,-93,44]  # home
        # point_d = [-6.5,-13.8,86.5,-4.5,-90.3,46]  #动作三
        point_A1 = [4, 21, 101, -36, -90, -40]
        point_A2 = [-16, 22, 102, -36, -90, -65]
        point_A3 = [-38, 28, 91, -32, -85, -80]

        point_c = [-28, -20, 105, 8.5, -90, -65]  # home
        point_d = [-26.7, -2, 65, -2.7, -90, -65]  # 动作三
        run_point(move, point_c)
        dashboard.ToolDOExecute(1, 0)
        dashboard.ToolDOExecute(2,1)

        generated_label = paradigm.paradigm()
        label = generated_label.split(": ")[1]  # 提取标签部分
        print(generated_label)
        print(label)


        if label == "左位置抓取":
            run_point(move, point_c)
            dashboard.ToolDOExecute(1, 0)
            dashboard.ToolDOExecute(2, 1)  # 开爪
            run_point(move, point_A1)  #左位置
            sleep(2)
            dashboard.ToolDOExecute(1, 1)
            dashboard.ToolDOExecute(2, 0)  #闭爪
            sleep(2)
            run_point(move, point_c)
            sleep(2)
            run_point(move, point_A1)
            dashboard.ToolDOExecute(1, 0)
            dashboard.ToolDOExecute(2, 1)  # 开爪
            sleep(2)
            run_point(move, point_c)

        elif label == "中位置抓取":
            run_point(move, point_c)
            dashboard.ToolDOExecute(1, 0)
            dashboard.ToolDOExecute(2, 1)  # 开爪
            run_point(move, point_A2)  # 中位置
            sleep(2)
            dashboard.ToolDOExecute(1, 1)
            dashboard.ToolDOExecute(2, 0)  # 闭爪
            sleep(2)
            run_point(move, point_c)
            sleep(2)
            run_point(move, point_A2)
            dashboard.ToolDOExecute(1, 0)
            dashboard.ToolDOExecute(2, 1)  # 开爪
            sleep(2)
            run_point(move, point_c)

        elif label == "右位置抓取":
            run_point(move, point_c)
            dashboard.ToolDOExecute(1, 0)
            dashboard.ToolDOExecute(2, 1)  # 开爪
            run_point(move, point_A3)  # 左位置
            sleep(2)
            dashboard.ToolDOExecute(1, 1)
            dashboard.ToolDOExecute(2, 0)  # 闭爪
            sleep(2)
            run_point(move, point_c)
            sleep(2)
            run_point(move, point_A3)
            dashboard.ToolDOExecute(1, 0)
            dashboard.ToolDOExecute(2, 1)  # 开爪
            sleep(2)
            run_point(move, point_c)
            # 每次运行间隔
        sleep(5)

if __name__ == "__main__":
    main()
#         neuracle = dict(device_name='Neuracle', hostname='127.0.0.1', port=8712,
#                         # srate=1000, chanlocs=['Pz', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'Oz', 'O1', 'O2', 'TRG'], n_chan=10)
#                         srate=1000, chanlocs=['C1', 'C2', 'C3', 'C4', 'TRG'], n_chan=65)
#
#         heeg = dict(device_name='HEEG', hostname='127.0.0.1', port=8172,
#                     srate=4000, chanlocs=['ch'] * 512, n_chan=513)
#
#         dsi = dict(device_name='DSI-24', hostname='127.0.0.1', port=8844,
#                    srate=300,
#                    chanlocs=['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'CM', 'A1', 'Fp1', 'Fp2', 'T3', 'T5', 'O1',
#                              'O2', 'X3', 'X2', 'F7', 'F8', 'X1', 'A2', 'T6', 'T4', 'TRG'], n_chan=25)
#         neuroscan = dict(device_name='Neuroscan', hostname='127.0.0.1', port=4000,
#                          srate=1000,
#                          chanlocs=['Pz', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'Oz', 'O1', 'O2', 'TRG'] + ['TRG'], n_chan=65)
#         device = [neuracle,heeg, dsi, neuroscan]
#         # 选着设备型号,默认Neuracle
#         target_device = device[0]
#         # 初始化 DataServerThread 线程
#         time_buffer = 3  # second
#         thread_data_server = DataServerThread(device=target_device['device_name'], n_chan=target_device['n_chan'],
#                                               srate=target_device['srate'], t_buffer=time_buffer)
#         # 建立TCP/IP连接
#         notconnect = thread_data_server.connect(hostname=target_device['hostname'], port=target_device['port'])
#         if notconnect:
#             raise TypeError("Can't connect recorder, Please open the hostport ")
#         else:
#             # 启动线程
#             thread_data_server.Daemon = True
#             thread_data_server.start()
#             print('Data server connected')
#         '''
#         在线数据获取演示：每隔一秒钟，获取数据（数据长度 = time_buffer）
#         '''
#         f = open('PLV.model', "rb")
#         rfc1 = pickle.load(f)
#         f.close()
#         MM = [0]
#         NN = [-1]
#         lastdata = np.empty((60, 0))
#         N, flagstop = 0, False
#         try:
#             while not flagstop:  # get data in one second step
#                 nUpdate = thread_data_server.GetDataLenCount()
#                 if nUpdate > (1 * target_device['srate'] - 1):
#                     N += 1
#                     data = thread_data_server.GetBufferData()
#                     # print('--------------------------------------------第几次采集：',N)
#                     thread_data_server.ResetDataLenCount()
#                     # print('直接接受到的数据', data.shape)
#                     olddata = data[0:59, :]
#                     # print('取前59通道的数据', olddata.shape)
#                     labeldata = data[[-1]]
#                     # print('取标签通道的数据', labeldata.shape)
#                     sumtnanone = np.sum(data[-1, :] > 0)
#                     whereihex,final = checklabel(int(sumtnanone), data[-1, :])
#                     time.sleep(1)
#                     # if int(final) == 1:
#                     #     run_point(move, point_c)
#                     #     time.sleep(2)
#                     #     run_point(move, point_a1)
#                     #     time.sleep(2)
#                     #     dashboard.ToolDOExecute(1, 1)
#                     #     dashboard.ToolDOExecute(2, 0)
#                     #     time.sleep(1)
#                     #     run_point(move, point_a2)
#                     #     time.sleep(1)
#                     #     run_point(move, point_a3)
#                     #     time.sleep(2)
#                     #     dashboard.ToolDOExecute(1, 0)
#                     #     dashboard.ToolDOExecute(2, 1)
#                     #     # wait_arrive(point_a)
#                     #
#                     # elif int(final) == 2:
#                     #     run_point(move, point_c)
#                     #     time.sleep(2)
#                     #     run_point(move, point_b1)
#                     #     time.sleep(2)
#                     #     dashboard.ToolDOExecute(1, 1)
#                     #     dashboard.ToolDOExecute(2, 0)
#                     #     time.sleep(1)
#                     #     run_point(move, point_b2)
#                     #     time.sleep(1)
#                     #     run_point(move, point_b3)
#                     #     time.sleep(2)
#                     #     dashboard.ToolDOExecute(1, 0)
#                     #     dashboard.ToolDOExecute(2, 1)
#                     # elif int(final) == 3:
#                     #     run_point(move, point_c)
#                     #     time.sleep(1)
#                     #     run_point(move, point_d)
#                     #     time.sleep(2)
#
#                     if int(final) == 1:
#                         run_point(move, point_c)
#                         dashboard.ToolDOExecute(1, 0)
#                         dashboard.ToolDOExecute(2, 1)  # 开抓
#                         run_point(move, point_a11)
#                         run_point(move, point_a1)  # 左抓
#                         time.sleep(3)
#                         dashboard.ToolDOExecute(1, 1)
#                         dashboard.ToolDOExecute(2, 0)  # 闭抓
#                         time.sleep(1)
#                         # run_point(move, point_a12)
#                         run_point(move, point_a2)
#                         run_point(move, point_a21)
#                         run_point(move, point_a3)
#                         time.sleep(3)
#                         dashboard.ToolDOExecute(1, 0)
#                         dashboard.ToolDOExecute(2, 1)
#                         time.sleep(0.5)
#                         run_point(move, point_a22)
#                         run_point(move, point_c)
#                         # wait_arrive(point_a)
#
#                     elif int(final) == 2:
#
#                         run_point(move, point_b11)
#                         run_point(move, point_b1)
#                         time.sleep(4)
#                         dashboard.ToolDOExecute(1, 1)
#                         dashboard.ToolDOExecute(2, 0)
#                         time.sleep(1)
#
#                         run_point(move, point_b2)
#                         run_point(move, point_b21)
#                         run_point(move, point_b3)
#                         time.sleep(3.5)
#                         dashboard.ToolDOExecute(1, 0)
#                         dashboard.ToolDOExecute(2, 1)
#                         time.sleep(0.5)
#                         run_point(move, point_b22)
#                         run_point(move, point_c)
#                         # wait_arrive(point_b)
#                     elif int(final) == 3:
#
#                         run_point(move, point_d)
#                         time.sleep(0.5)
#                         run_point(move, point_c)
#
#                 if N > 100:
#                     flagstop = True
#         except:
#             pass
#         # 结束线程
#         thread_data_server.stop()
#
# if __name__ == '__main__':
#
#
#    main()