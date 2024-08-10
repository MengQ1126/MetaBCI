import math
import numpy as np
import warnings
import threading
from time import sleep
from psychopy import monitors
from psychopy.tools.monitorunittools import deg2pix
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType
from metabci.brainstim.paradigm import (
    SSVEP,
    paradigm,
    generate_random_label,
)
from metabci.brainstim.framework import (Experiment,)
from neuracle_lib.dataServer import DataServerThread

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
            current_actual = a["tool_vector_actual"][0]
        sleep(0.001)

def wait_arrive(point_list):
    global current_actual
    while True:
        is_arrive = True
        if current_actual is not None:
            for index in range(len(current_actual)):
                if abs(current_actual[index] - point_list[index]) > 1:
                    is_arrive = False
            if is_arrive:
                return
        sleep(0.001)

# def main():


if __name__ == "__main__":
    # main()
    # SSVEP experiment setup
    mon = monitors.Monitor(
        name="primary_monitor",
        width=59.6,
        distance=60,
        verbose=False,
    )
    mon.setSizePix([1460, 920])
    mon.save()

    bg_color_warm = np.array([0, 0, 0])
    win_size = np.array([1460, 920])

    ex = Experiment(
        monitor=mon,
        bg_color_warm=bg_color_warm,
        screen_id=0,
        win_size=win_size,
        is_fullscr=False,
        record_frames=False,
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    )
    win = ex.get_window()

    n_elements, rows, columns = 3, 1, 3
    stim_length, stim_width = 300, 300
    stim_color, tex_color = [1, 1, 1], [1, 1, 1]
    fps = 240
    stim_time = 2
    stim_opacities = 1
    freqs = np.arange(8, 16, 0.4)
    phases = np.array([i * 0.35 % 2 for i in range(n_elements)])

    basic_ssvep = SSVEP(win=win)
    basic_ssvep.config_pos(n_elements=n_elements, rows=rows, columns=columns, stim_length=stim_length,
                           stim_width=stim_width)
    basic_ssvep.config_text(tex_color=tex_color)
    basic_ssvep.config_color(refresh_rate=fps, stim_time=stim_time, stimtype="sinusoid", stim_color=stim_color,
                             stim_opacities=stim_opacities, freqs=freqs, phases=phases)
    basic_ssvep.config_index()
    basic_ssvep.config_response()

    bg_color = np.array([0.3, 0.3, 0.3])
    display_time = 1
    index_time = 1
    rest_time = 0.5
    response_time = 1
    port_addr = None
    nrep = 2
    lsl_source_id = "meta_online_worker"
    online = False

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
    # ex.run()
    # input("Press Enter to generate the label and proceed with the next steps...")
    # generated_label = paradigm.paeadigm()
    # sleep(80)
    # generated_label = generate_random_label()
    generated_label = generate_random_label()
    label = generated_label
    print("执行的操作是：",label)
    dashboard, move, feed = connect_robot()
    print("开始上电...")
    dashboard.PowerOn()
    # y = int(input("已上电，使能请按1，否则退出:  "))
    # if y == 1:
    print("开始使能...")
    dashboard.EnableRobot()
    print("完成使能:)")
    feed_thread = threading.Thread(target=get_feed, args=(feed,))
    feed_thread.setDaemon(True)
    feed_thread.start()

    point_A1 = [4, 21, 101, -36, -90, -40]
    point_A2 = [-16, 22, 102, -36, -90, -65]
    point_A3 = [-38, 28, 91, -32, -85, -80]
    point_B = [-28, -20, 105, 8.5, -90, -65]

    run_point(move, point_B)
    dashboard.ToolDOExecute(1, 0)
    dashboard.ToolDOExecute(2, 1)

    if label == "左位置抓取":
        run_point(move, point_B)
        dashboard.ToolDOExecute(1, 0)
        dashboard.ToolDOExecute(2, 1)
        run_point(move, point_A1)
        sleep(2)
        dashboard.ToolDOExecute(1, 1)
        dashboard.ToolDOExecute(2, 0)
        sleep(2)
        run_point(move, point_B)
        sleep(2)
        run_point(move, point_A1)
        sleep(2)
        dashboard.ToolDOExecute(1, 0)
        dashboard.ToolDOExecute(2, 1)
        sleep(2)
        run_point(move, point_B)

    elif label == "中位置抓取":
        run_point(move, point_B)
        dashboard.ToolDOExecute(1, 0)
        dashboard.ToolDOExecute(2, 1)
        run_point(move, point_A2)
        sleep(2)
        dashboard.ToolDOExecute(1, 1)
        dashboard.ToolDOExecute(2, 0)
        sleep(2)
        run_point(move, point_B)
        sleep(2)
        run_point(move, point_A2)
        sleep(2)
        dashboard.ToolDOExecute(1, 0)
        dashboard.ToolDOExecute(2, 1)
        sleep(2)
        run_point(move, point_B)

    elif label == "右位置抓取":
        run_point(move, point_B)
        dashboard.ToolDOExecute(1, 0)
        dashboard.ToolDOExecute(2, 1)
        run_point(move, point_A3)
        sleep(2)
        dashboard.ToolDOExecute(1, 1)
        dashboard.ToolDOExecute(2, 0)
        sleep(2)
        run_point(move, point_B)
        sleep(2)
        run_point(move, point_A3)
        sleep(2)
        dashboard.ToolDOExecute(1, 0)
        dashboard.ToolDOExecute(2, 1)
        sleep(2)
        run_point(move, point_B)

    sleep(5)
ex.run()


