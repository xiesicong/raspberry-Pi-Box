"""通过网络编程遥控，同时传递图像。PC服务端"""
import cv2
import socket
import numpy
import struct
import threading
import time

import human_detect
import face_detect


class Setting:
	def __init__(self):
		#self.picture_port = (socket.gethostname(), 8080)
		#self.control_port = (socket.gethostname(), 8081)
		#self.picture_port = ('240e:ba:d00f:a87:64df:4726:36cb:5d66', 8080)
		#self.control_port = ('240e:ba:d00f:a87:64df:4726:36cb:5d66', 8081)
		self.picture_port = ('192.168.43.110', 8080)
		self.control_port = ('192.168.43.110', 8081)
		self.picture_size = (640, 480)  # (480, 360)  # (960, 720)
		self.buffsize = 65535


class Status:
	def __init__(self):
		self.image = None
		self.human_detect_image_new = ''
		self.human_flag = False
		self.face_name = 'unknown'
		self.re_flag = False


def recv_picture(setting, status):
	picture_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	picture_socket.bind(setting.picture_port)
	while True:
		data, address = picture_socket.recvfrom(setting.buffsize)
		if len(data) == 1 and data[0] == 1:  # 如果收到关闭消息则停止程序
			picture_socket.close()
			cv2.destroyAllWindows()
			exit()
		if len(data) != 4:  # 进行简单的校验，长度值是int类型，占四个字节
			length = 0
		else:
			length = struct.unpack('i', data)[0]  # 长度值
		data, address = picture_socket.recvfrom(setting.buffsize)  # 接收编码图像数据
		if length != len(data):  # 进行简单的校验
			continue
		data = numpy.array(bytearray(data))  # 格式转换
		image = cv2.imdecode(data, 1)  # 图像解码
		image = cv2.resize(image, setting.picture_size, interpolation=cv2.INTER_CUBIC)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		status.image = image


def main():
	setting = Setting()
	status = Status()

	recv_picture_thread = threading.Thread(target=recv_picture, args=(setting, status))
	recv_picture_thread.start()

	control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	control_socket.bind(setting.control_port)
	control_socket.listen(2)
	while True:
		#   接收客户端连接
		print("等待控制连接....")
		client, address = control_socket.accept()
		print("新控制连接")
		print("IP is %s" % address[0])
		print("port is %d\n" % address[1])

		time.sleep(5)
		threading.Thread(target=human_detect.main, args=(status,)).start()  # 行人检测并更新显示的线程
		threading.Thread(target=face_detect.main, args=(status,)).start()  # 人脸识别并更新显示的线程
		time.sleep(10)
		while True:
			#cv2.imshow('human detect', cv2.cvtColor(numpy.asarray(status.human_detect_image_new), cv2.COLOR_RGB2BGR))
			# 发消息
			if status.human_flag:  # 有人
				client.send("有人".encode("utf-8"))
			else:  # 没人
				client.send("没人".encode("utf-8"))

			if status.re_flag:
				client.send("restart".encode("utf-8"))
				print("出错")
				break

			time.sleep(0.1)


if __name__ == '__main__':
	main()
