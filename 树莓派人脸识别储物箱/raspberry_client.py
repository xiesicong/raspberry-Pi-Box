#-*-coding:utf-8-*
"""行人检测云计算。树莓派客户端"""
import os
import RPi.GPIO as GPIO
import time
import threading
import socket
import struct
import cv2

from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.legacy import text
from luma.core.legacy.font import proportional, LCD_FONT


def main():
	# 初始化们
	setting = Setting()
	status = Status()
	GPIO_init(setting)
	# 初始化max7219显示屏
	serial = spi(port=0, device=0, gpio=noop())
	status.device = max7219(serial, cascaded=3, block_orientation=0)
	max7219_display('init', status.device)
	# 初始化三个子线程并运行
	# 检测开关的线程
	trigger_thread = threading.Thread(target=trigger_rising, args=(setting,))
	trigger_thread.start()
	# 控制灯的线程
	light_control_thread = threading.Thread(target=light_control, args=(setting,))
	light_control_thread.start()
	# 发送图像的线程
	send_picture_thread = threading.Thread(target=send_picture, args=(setting,))
	send_picture_thread.start()

	control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	control_socket.connect(setting.control_socket_ip_port)
	print("控制通道连接成功\n")

	print("开始行人检测")
	max7219_display('start', status.device)
	try:
		while setting.program_running_flag:
			check_event_in_main_loop(status, control_socket)
			response_event(setting, status)
			time.sleep(0.1)
	except KeyboardInterrupt:
		setting.program_running_flag = False
		GPIO.cleanup()
		os._exit(0)
		print('退出程序')


class Setting:

	def __init__(self):
		self.light_vcc = 15  #26  # led灯的正极
		self.trigger_vcc = 12  # 脉冲开关的一边，随意
		self.trigger_gnd = 6  # 脉冲开关的另一边
		self.stm32 = 21  # 给stm32传递信号，有人就是低电平，没人就是高电平

		self.light_flag = False  # 是否亮灯的标志
		self.lighting_time = 3  # 强行打开灯持续的时间
		self.detect_flag = True  # 是否要执行检测程序
		self.rising_time = time.time()  # 记录脉冲开关接通的时刻
		self.program_running_flag = True  # 是否结束子线程的标志


		#self.picture_socket_ip_port = ('chenshouyang.top', 7001)
		#self.control_socket_ip_port = ('chenshouyang.top', 7011)
		#self.picture_socket_ip_port = ('240e:ba:d00f:a87:64df:4726:36cb:5d66', 8080)
		#self.control_socket_ip_port = ('240e:ba:d00f:a87:64df:4726:36cb:5d66', 8081)
		self.picture_socket_ip_port = ('192.168.43.110', 8080)
		self.control_socket_ip_port = ('192.168.43.110', 8081)
		self.picture_size = (480, 360)
		self.picture_fps = 50


class Status:
	def __init__(self):
		self.human_detect_result = False
		self.face_name = ''
		self.device = None  # 点阵屏对象


def GPIO_init(setting):
	"""
	初始化GPIO口
	:param setting:
	:return:
	"""
	GPIO.setmode(GPIO.BCM)
	GPIO.setwarnings(False)
	GPIO.setup(setting.light_vcc, GPIO.OUT, initial=GPIO.LOW)
	GPIO.setup(setting.trigger_vcc, GPIO.OUT, initial=GPIO.HIGH)
	GPIO.setup(setting.trigger_gnd, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
	GPIO.setup(setting.stm32, GPIO.OUT, initial=GPIO.HIGH)


def trigger_rising(setting):
	"""
	检测脉冲开关的线程
	:param setting:
	:return:
	"""
	GPIO.add_event_detect(setting.trigger_gnd, GPIO.RISING)
	while setting.program_running_flag:
		if GPIO.event_detected(setting.trigger_gnd):
			setting.light_flag = True
			setting.detect_flag = False
			setting.rising_time = time.time()
		if time.time() - setting.rising_time > setting.lighting_time:
			setting.detect_flag = True
		time.sleep(0.1)


def light_control(setting):
	"""
	根据变量控制灯通断的线程
	:param setting:
	:return:
	"""
	while setting.program_running_flag:
		if setting.light_flag:
			GPIO.output(setting.light_vcc, GPIO.HIGH)
			GPIO.output(setting.stm32, GPIO.LOW)
		else:
			if time.time() - setting.rising_time > setting.lighting_time:
				# 加一个保护，如果离脉冲开关接通还没有到规定的时间那就不要关灯
				GPIO.output(setting.light_vcc, GPIO.LOW)
				GPIO.output(setting.stm32, GPIO.HIGH)
		time.sleep(0.1)


def send_picture(setting):
	picture_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	picture_socket.connect(setting.picture_socket_ip_port)
	print("图像传输通道连接成功\n")
	camera = cv2.VideoCapture(0)
	img_param = [cv2.IMWRITE_JPEG_QUALITY, setting.picture_fps]
	while True:
		ret, frame = camera.read()
		if ret:
			frame = cv2.resize(frame, setting.picture_size)
			_, frame_encode = cv2.imencode('.jpg', frame, img_param)  # 按格式生成图片
			#try:
			picture_socket.sendall(struct.pack('i', frame_encode.shape[0]))  # 发送编码后的字节长度，这个值不是固定的
			picture_socket.sendall(frame_encode)  # 发送视频帧数据
			#except:
				#camera.release()  # 释放资源
				#return


def max7219_display(word, device):
	"""
	控制max7219的led矩阵屏幕显示内容，只支持ASCII字符
	:param word:
	:param device:
	:return:
	"""
	with canvas(device) as draw:
		text(draw, (0, 0), word, fill="white", font=proportional(LCD_FONT))


def check_event_in_main_loop(status, control_socket):
	msg = control_socket.recv(512).decode("utf-8")
	#print('接收完成')
	if msg == "有人":
		status.human_detect_result = True
	else:  # 没人
		status.human_detect_result = False


def response_event(setting, status):
	if setting.detect_flag:
		if status.human_detect_result:
			max7219_display('True', status.device)
			#print("有人")
		else:
			max7219_display('False', status.device)
			#print("没人")
			setting.light_flag = False
	else:
		# time.sleep(1)
		pass


if __name__ == '__main__':
	main()

