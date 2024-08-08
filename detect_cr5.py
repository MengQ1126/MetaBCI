<<<<<<< HEAD
import time
import cv2
import numpy as np
import torch
import copy
from models.common import DetectMultiBackend
from utils.general import check_img_size,non_max_suppression
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox #调整图片大小至640

from cam_calibration import HandInEyeCalibration
from CR_robot import CR_ROBOT, CR_bringup
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType

# 传输过来的分类结果
bci_data = 77

# 放置的位置
place_position = [-650, -200, 250]
# 初始位置
home_pose = [-300, 0, 510, 180, 0, 45]
# x，y是变量，只用z，rx, ry, rz
height_pose = [-400, 0, 200, 179, 1, 140]


if __name__ == '__main__':

    tcp_host_ip = '192.168.5.1'  # IP and port to robot arm as TCP client (UR5)
    tcp_port = 30003
    tcp_port2 = 29999

    robot = CR_ROBOT(tcp_host_ip, tcp_port)
    robot_init = CR_bringup(tcp_host_ip, tcp_port2)
    robot_init.EnableRobot()
    # robot.MovJ(int(home_pose[0]), int(home_pose[1]), int(home_pose[2]),
    #            home_pose[3], home_pose[4], home_pose[5])
    # Load model
    device = select_device('')
    weights = "yolov5s.pt"
    dnn = False
    data = "data/coco128.yaml"
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size((640, 640), s=stride)  # check image size

    model.warmup()  # warmup

    capture = cv2.VideoCapture(1)

    # 延迟300毫秒
    time.sleep(0.3)

    ret, frame = capture.read()  # 获取一帧图像
    frame = cv2.flip(frame, 1)  # 图像翻转

    img0 = frame
    img = letterbox(frame)[0]  # 返回的是元组所以[0]
    # img = frame.transpose((2, 0, 1))  # HWC to CHW
    img = frame.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)  # 转为连续数组
    im = torch.from_numpy(img).to(device)
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # 检测图像
    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred)
    det = pred[0]

    det2 = det.tolist()
    # print(det2)
    # 提取每行的最后一个元素放入新列表,并改为int类型
    number_list = list(map(int, [row[-1] for row in det2]))
    print(number_list)

    # 查找bci_data是否与det2每一行的最后一列相同
    row_index = None
    for i, row in enumerate(det2):
        if bci_data == row[-1]:
            row_index = i
            break

    annotator = Annotator(frame, line_width=3, example=str(names))
    for *xyxy, conf, cls in iter(det):
        c = int(cls)
        label = names[c]
        annotator.box_label(xyxy, label, color=colors(c, True))
    im0 = annotator.result()
    cv2.imshow('frame', im0)

    # 图像等待3000后关闭
    cv2.waitKey(30)

    if row_index is not None:
        x1, y1, x2, y2 = det2[row_index][:4]  # 获取每一行的前四列数据
        x_camera, y_camera = int((x1 + x2) / 2), int((y1 + y2) / 2)  # 获取所抓取的像素中心点坐标
        print("中间点坐标：({}, {})".format(x_camera, y_camera))
        # 创建 HandInEyeCalibration 实例对象
        calibrator = HandInEyeCalibration()
        # 调用 get_points_robot 方法，求解机械臂坐标
        robot_x, robot_y = calibrator.get_points_robot(x_camera, y_camera)
        print("机械臂坐标 (x, y):", robot_x, robot_y)

        '''-------------------抓取物体------------'''
        robot.MovJ(int(home_pose[0]), int(home_pose[1]), int(home_pose[2]),
                   home_pose[3], home_pose[4], home_pose[5])
        time.sleep(1)
        # 物体上方10cm
        robot.MovJ(int(robot_x), int(robot_y), int(height_pose[2] + 100),
                   height_pose[3], height_pose[4], height_pose[5])
        time.sleep(2)
        # 抓物体 可更改偏移量
        robot.MovJ(int(robot_x), int(robot_y), int(height_pose[2]),
                   height_pose[3], height_pose[4], height_pose[5])
        time.sleep(2)
        robot_init.ToolDOExecute(1, 1)
        robot_init.ToolDOExecute(2, 0)
        # robot_init.ToolDOExecute(2, 0)
        time.sleep(1)
        robot.MovJ(int(robot_x), int(robot_y)+30, int(height_pose[2] + 100),
                   height_pose[3], height_pose[4], height_pose[5])
        time.sleep(1)
        robot.MovJ(int(place_position[0]), int(place_position[1]), int(place_position[2]),
                   height_pose[3], height_pose[4], height_pose[5])
        time.sleep(2)
        robot_init.ToolDOExecute(1, 0)
        robot_init.ToolDOExecute(2, 1)
        # robot_init.ToolDOExecute(2, 1)
        time.sleep(2)
        robot.MovJ(int(home_pose[0]), int(home_pose[1]), int(home_pose[2]),
                   home_pose[3], home_pose[4], home_pose[5])
        # time.sleep(2)
        '''------------------------------------------------------------------------------'''

    else:
        print("BCI data not found in any row.")

    capture.release()
    cv2.destroyAllWindows()
=======
import time
import cv2
import numpy as np
import torch
import copy
from models.common import DetectMultiBackend
from utils.general import check_img_size,non_max_suppression
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox #调整图片大小至640

from cam_calibration import HandInEyeCalibration
from CR_robot import CR_ROBOT, CR_bringup
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType

# 传输过来的分类结果
bci_data = 77

# 放置的位置
place_position = [-650, -200, 250]
# 初始位置
home_pose = [-300, 0, 510, 180, 0, 45]
# x，y是变量，只用z，rx, ry, rz
height_pose = [-400, 0, 200, 179, 1, 140]


if __name__ == '__main__':

    tcp_host_ip = '192.168.5.1'  # IP and port to robot arm as TCP client (UR5)
    tcp_port = 30003
    tcp_port2 = 29999

    robot = CR_ROBOT(tcp_host_ip, tcp_port)
    robot_init = CR_bringup(tcp_host_ip, tcp_port2)
    robot_init.EnableRobot()
    # robot.MovJ(int(home_pose[0]), int(home_pose[1]), int(home_pose[2]),
    #            home_pose[3], home_pose[4], home_pose[5])
    # Load model
    device = select_device('')
    weights = "yolov5s.pt"
    dnn = False
    data = "data/coco128.yaml"
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size((640, 640), s=stride)  # check image size

    model.warmup()  # warmup

    capture = cv2.VideoCapture(1)

    # 延迟300毫秒
    time.sleep(0.3)

    ret, frame = capture.read()  # 获取一帧图像
    frame = cv2.flip(frame, 1)  # 图像翻转

    img0 = frame
    img = letterbox(frame)[0]  # 返回的是元组所以[0]
    # img = frame.transpose((2, 0, 1))  # HWC to CHW
    img = frame.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)  # 转为连续数组
    im = torch.from_numpy(img).to(device)
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # 检测图像
    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred)
    det = pred[0]

    det2 = det.tolist()
    # print(det2)
    # 提取每行的最后一个元素放入新列表,并改为int类型
    number_list = list(map(int, [row[-1] for row in det2]))
    print(number_list)

    # 查找bci_data是否与det2每一行的最后一列相同
    row_index = None
    for i, row in enumerate(det2):
        if bci_data == row[-1]:
            row_index = i
            break

    annotator = Annotator(frame, line_width=3, example=str(names))
    for *xyxy, conf, cls in iter(det):
        c = int(cls)
        label = names[c]
        annotator.box_label(xyxy, label, color=colors(c, True))
    im0 = annotator.result()
    cv2.imshow('frame', im0)

    # 图像等待3000后关闭
    cv2.waitKey(30)

    if row_index is not None:
        x1, y1, x2, y2 = det2[row_index][:4]  # 获取每一行的前四列数据
        x_camera, y_camera = int((x1 + x2) / 2), int((y1 + y2) / 2)  # 获取所抓取的像素中心点坐标
        print("中间点坐标：({}, {})".format(x_camera, y_camera))
        # 创建 HandInEyeCalibration 实例对象
        calibrator = HandInEyeCalibration()
        # 调用 get_points_robot 方法，求解机械臂坐标
        robot_x, robot_y = calibrator.get_points_robot(x_camera, y_camera)
        print("机械臂坐标 (x, y):", robot_x, robot_y)

        '''-------------------抓取物体------------'''
        robot.MovJ(int(home_pose[0]), int(home_pose[1]), int(home_pose[2]),
                   home_pose[3], home_pose[4], home_pose[5])
        time.sleep(1)
        # 物体上方10cm
        robot.MovJ(int(robot_x), int(robot_y), int(height_pose[2] + 100),
                   height_pose[3], height_pose[4], height_pose[5])
        time.sleep(2)
        # 抓物体 可更改偏移量
        robot.MovJ(int(robot_x), int(robot_y), int(height_pose[2]),
                   height_pose[3], height_pose[4], height_pose[5])
        time.sleep(2)
        robot_init.ToolDOExecute(1, 1)
        robot_init.ToolDOExecute(2, 0)
        # robot_init.ToolDOExecute(2, 0)
        time.sleep(1)
        robot.MovJ(int(robot_x), int(robot_y)+30, int(height_pose[2] + 100),
                   height_pose[3], height_pose[4], height_pose[5])
        time.sleep(1)
        robot.MovJ(int(place_position[0]), int(place_position[1]), int(place_position[2]),
                   height_pose[3], height_pose[4], height_pose[5])
        time.sleep(2)
        robot_init.ToolDOExecute(1, 0)
        robot_init.ToolDOExecute(2, 1)
        # robot_init.ToolDOExecute(2, 1)
        time.sleep(2)
        robot.MovJ(int(home_pose[0]), int(home_pose[1]), int(home_pose[2]),
                   home_pose[3], home_pose[4], home_pose[5])
        # time.sleep(2)
        '''------------------------------------------------------------------------------'''

    else:
        print("BCI data not found in any row.")

    capture.release()
    cv2.destroyAllWindows()
>>>>>>> 40a1b26e5531c86619ddd8523efb39e3f286cc6a
