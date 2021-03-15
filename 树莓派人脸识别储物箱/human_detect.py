"""行人检测云计算。PC服务端"""
from imageai.Detection import ObjectDetection
import time
import os
import cv2


class Setting:
	def __init__(self):
		execution_path = os.getcwd()
		# self.image_path = os.path.join(execution_path, 'image.jpg')  # '/home/pi/ImageAI/image.jpg'
		# self.image_new_path = os.path.join(execution_path, 'image_new.jpg')  # '/home/pi/ImageAI/imagenew.jpg'
		self.train_data_path = os.path.join(execution_path, 'human_detect\\resnet50_coco_best_v2.0.1.h5')  # '/home/pi/ImageAI/resnet50_coco_best_v2.0.1.h5'


def main(status):
	# 初始化们
	setting = Setting()

	# 初始化神经网络
	detector = ObjectDetection()
	detector.setModelTypeAsRetinaNet()
	detector.setModelPath(setting.train_data_path)
	detector.loadModel(detection_speed='fast')  # 可用的检测速度是“normal”(default),“fast”,“faster” ,“fastest”and“flash”
	custom_objects = detector.CustomObjects(person=True)

	print("开始行人检测")
	#time1 = time.time()  # 记录开始时时间点
	#try:
	while True:
		detected_image_array, detections = detector.detectCustomObjectsFromImage(
			input_type="array",
			output_type="array",
			custom_objects=custom_objects,
			input_image=status.image,
			#output_image_path=status.human_detect_image_new
		)
		if len(detections) > 0:
			#print('有人', end='')
			status.human_flag = True
		else:
			#print('没人', end='')
			status.human_flag = False
		#time2 = time.time()
		#print(',此帧花费{}秒'.format(time2 - time1))
		cv2.imshow('human detect', cv2.cvtColor(detected_image_array, cv2.COLOR_BGR2RGB))
		#cv2.imshow('result', status.human_detect_image_new)
		#time1 = time2

		# 按Q退出
		if cv2.waitKey(1) & 0xFF == ord('q'):
			status.re_flag = True
			break
	#except:
		#status.re_flag = True

