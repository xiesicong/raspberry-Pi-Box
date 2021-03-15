# -*- coding: utf-8 -*-
# 摄像头头像识别
"""
只能标注非中文的名字
没有多线程
"""
import face_recognition
import cv2
import json
import numpy
from PIL import Image
import time
import sys
import os


class People:
    def __init__(self, recognized_flag=False):
        self.name = ''
        self.root_dir = ''
        self.image_names = []
        self.encodings = []

        self.recognized_flag = recognized_flag

    def append_name_and_encoding(self, name):
        """
        接收绝对文件路径，把图片以及图片编码分别添加到属性里面去
        :param name:
        :return:
        """
        try:
            self.image_names.append(name)
            image = face_recognition.load_image_file(name)
            encoding = face_recognition.face_encodings(image)[0]
            self.encodings.append(encoding)
            return True
        except IndexError:
            print(name+' 找不到人脸')
            self.image_names.remove(name)
            os.remove(name)
            return False


class Setting:
    def __init__(self):
        self.recognized_root = 'face_detect/recognized'
        self.unrecognized_root = 'face_detect/unrecognized'
        self.files_root = 'face_detect/files'
        self.tolerance_value = 0.37  # 亚洲人时改到0.37  # 默认白人0.6


class Status:
    def __init__(self):
        self.recognized_people_list = []
        self.unrecognized_people_list = []
        self.last_name = ''


def load_status(setting, status):
    """
    把文件中的status加载到内存中。同样是分开加载的。
    :param setting:
    :param status:
    :return:
    """
    try:
        with open(setting.files_root + '/recognized_name_list.json', 'r') as f1:
            with open(setting.files_root + '/recognized_root_dir_list.json', 'r') as f2:
                with open(setting.files_root + '/recognized_image_names_list.json', 'r') as f3:
                    with open(setting.files_root + '/recognized_recognized_flag_list.json', 'r') as f5:
                        name_list = json.load(f1)
                        root_dir_list = json.load(f2)
                        image_names_list = json.load(f3)
                        encodings_list = numpy.load(setting.files_root + '/recognized_encodings_list.npy')
                        recognized_flag_list = json.load(f5)
                        for number in range(len(name_list)):
                            people = People(recognized_flag=True)
                            people.name = name_list[number]
                            people.root_dir = root_dir_list[number]
                            people.image_names = image_names_list[number]
                            people.encodings = encodings_list[number]
                            people.recognized_flag = recognized_flag_list[number]
                            status.recognized_people_list.append(people)
                        print('已知面孔加载完毕')
    except FileNotFoundError:
        print('没有找到缓存,正在重新加载图片')
        return None


def cut_and_save(name, status, frame, box):
    """
    把已经找到了的人脸加入到recognized的文件夹
    :param name:
    :param status:
    :param frame:
    :param box:
    :return:
    """
    for people in status.recognized_people_list:
        if name == people.name:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            region = pil_img.crop(box)
            #print(people.name)
            #print(people.root_dir)
            if name != status.last_name:
                region.save(str(people.root_dir) + '\\' + str(int(time.time())) + '.BMP')
            status.last_name = name


def calculate_picture(picture_status, process_this_frame, known_face_encodings, known_face_names, face_locations,
                      face_names, status, setting):
    """
    不断地循环，一帧一帧的读取视频，然后处理
    :param picture_status:
    :param process_this_frame:
    :param known_face_encodings:
    :param known_face_names:
    :param face_locations:
    :param face_names:
    :param status:
    :return:
    """
    while True:
        frame = picture_status.image.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 改变摄像头图像的大小，图像小，所做的计算就少
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # opencv的图像是BGR格式的，而我们需要是的RGB格式的，因此需要进行一个转换。
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # 根据encoding来判断是不是同一个人，是就输出true，不是为flase
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # 默认为unknown
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=setting.tolerance_value)
                name = "Unknown"

                # if match[0]:
                #     name = "michong"
                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                picture_status.face_name = name  # 返回识别到的人的名字
                face_names.append(name)

        process_this_frame = not process_this_frame

        # 将捕捉到的人脸显示出来
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            """
            top = int(top * 3.8)
            right *= 4
            bottom = int(bottom * 4.2)
            left *= 4
            """

            # 把找到的人脸添加到已知的文件夹，增加下一次识别成功率
            if name != "Unknown":
                cut_and_save(name, status, frame, (left, top, right, bottom))

            # 矩形框
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # 加上标签
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display
        cv2.imshow('face detect', frame)
        time.sleep(0.1)

        # 按Q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            picture_status.re_flag = True
            break


def connect_absolute_path(setting):
    """
    将传递进来的相对文件路径，变成绝对文件路径（不包含文件）
    :param setting:
    :return:
    """
    setting.recognized_root = os.path.abspath(setting.recognized_root)
    setting.unrecognized_root = os.path.abspath(setting.unrecognized_root)


def check_this_file_appended_recognized(root, file, status):
    """
    检查在文件夹遍历中的当前文件，是否已经在已经识别了的列表中。如果没有则加进去
    :param root: 不包含文件的绝对文件路径
    :param file: 不包含路径的文件名
    :param status:
    :return:
    """
    list1 = root.split('\\')
    people_name = list1[-1]
    file_name = os.path.join(root, file)
    find_out_flag = False  # 是否在接下来的遍历中被找到
    for people in status.recognized_people_list:
        if people_name == people.name:  # 判断绝对文件路径的最后一项文件夹，也就是人的名字，是否在已知的人当中。有则代表被找到
            find_out_flag = True
            if file_name not in people.image_names:  # 当前文件是否在这个人下面已知的图片中，如果不是则加进去。
                people.append_name_and_encoding(file_name)
            break
    if not find_out_flag:  # 遍历完了都没找到这个人，则把这个人加入到认识的人的列表中。
        new_people = People(recognized_flag=True)
        new_people.name = people_name
        new_people.root_dir = root
        new_people.append_name_and_encoding(file_name)
        status.recognized_people_list.append(new_people)


def traversing_recognized(root_dir, status):
    """
    遍历已知人的文件夹，并且加入到内存中。
    :param root_dir: 不带文件的绝对文件路径
    :param status:
    :return:
    """
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if '.' in file:
                check_this_file_appended_recognized(root, file, status)
        for dir in dirs:
            traversing_recognized(dir, status)


def save_status(setting, status):
    """
    将status中的两个列表存入到文件中。但是json不认识自己写的类，所以只好分开储存。encoding要用numpy的储存和读取。
    :param setting:
    :param status:
    :return:
    """
    with open(setting.files_root + '/recognized_name_list.json', 'w') as f1:
        with open(setting.files_root + '/recognized_root_dir_list.json', 'w') as f2:
            with open(setting.files_root + '/recognized_image_names_list.json', 'w') as f3:
                with open(setting.files_root + '/recognized_recognized_flag_list.json', 'w') as f5:
                    name_list = []
                    root_dir_list = []
                    image_names_list = []
                    encodings_list = []
                    recognized_flag_list = []
                    for people in status.recognized_people_list:
                        name_list.append(people.name)
                        root_dir_list.append(people.root_dir)
                        image_names_list.append(people.image_names)
                        encodings_list.append(people.encodings)
                        recognized_flag_list.append(people.recognized_flag)
                    json.dump(name_list, f1)
                    json.dump(root_dir_list, f2)
                    json.dump(image_names_list, f3)
                    numpy.save(setting.files_root + '/recognized_encodings_list.npy', encodings_list)
                    json.dump(recognized_flag_list, f5)

                    print('已知面孔储存完毕')


def cut_people_face(status):
    """
    把已认识的人文件夹里面不是BMP文件的全部把头剪下来，储存为BMP文件
    :param status:
    :return:
    """
    # 使用默认的给予HOG模型查找图像中所有人脸
    # 这个方法已经相当准确了，但还是不如CNN模型那么准确，因为没有使用GPU加速
    # 另请参见: find_faces_in_picture_cnn.py
    for people in status.recognized_people_list:
        for image_name in people.image_names.copy():
            image_name_list = image_name.split('.')
            if image_name_list[-1] == 'BMP':
                continue
            else:
                unbmp_index = people.image_names.index(image_name)
                people.image_names.remove(image_name)
                people.encodings.pop(unbmp_index)
            image = face_recognition.load_image_file(image_name)
            face_locations = face_recognition.face_locations(image)

            # 使用CNN模型
            # face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

            # 打印：我从图片中找到了 多少 张人脸
            print("I found {} face(s) in this photograph.".format(len(face_locations)))

            # 循环找到的所有人脸
            for face_location in face_locations:

                # 打印每张脸的位置信息
                top, right, bottom, left = face_location
                top = int(top * 0.9)
                right = int(right * 1.1)
                bottom = int(bottom * 1.1)
                left = int(left * 0.9)
                print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left,
                                                                                                            bottom,
                                                                                                            right))
                # 指定人脸的位置信息，然后显示人脸图片
                face_image = image[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                if face_locations.index(face_location) == 0:
                    image_name_list = image_name.split('.')
                    image_name_list[-1] = 'BMP'
                    image_name_new = image_name_list[0] + '.' + image_name_list[-1]
                    pil_image.save(image_name_new)
                    os.remove(image_name)


def main(picture_status):
    sys.setrecursionlimit(1000000)  # 修改递归深度限制
    setting = Setting()  # 初始化设置类
    status = Status()  # 初始化全局变量类
    load_status(setting, status)  # 先尝试读取文件中的status，如果是第一次运行，则这步不执行

    connect_absolute_path(setting)  # 把相对文件路径连接成绝对文件路径，避免后续出错
    traversing_recognized(setting.recognized_root, status)  # 编码新图片，并加入到已知列表
    print("编码已知图片成功")

    cut_people_face(status)  # 把人脸切下来

    #  建立已知人的编码和名字的列表
    known_face_encodings = []
    known_face_names = []
    for people in status.recognized_people_list:
        try:
            known_face_encodings.append(people.encodings[0])
        except IndexError:
            pass
        known_face_names.append(people.name)

    face_locations = []
    face_names = []
    process_this_frame = True

    # 计算：
    #try:
    calculate_picture(picture_status, process_this_frame, known_face_encodings, known_face_names, face_locations,
                      face_names, status, setting)
    #except:
        #picture_status.re_flag = True

    cv2.destroyAllWindows()

    # 储存status
    save_status(setting, status)  # 把已经认识了的编码之类的存下来，免得每次重新编码耽搁时间
