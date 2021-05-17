import os
import cv2

project_path = os.getcwd()
train_path = project_path + "/train/"


with open("train.csv", mode="w") as f:
    f.close()
    # 读取测试集图片数据
for index in range(80):
    train_dir = os.listdir(train_path + str(index) + "/")
    for file in train_dir:
        # 拼接路径便于读取
        img_path = train_path + str(index) + "/" + file
        print(img_path)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 输入的图像必须是float32格式的，最后一个参数的要求在上面
        dst = cv2.cornerHarris(gray, 3, 3, 0.06)
        # cv2.cornerHarris(src, blockSize, ksize, k[, dst[, borderType]])
        # src是输入图象
        # blocksize角点检测中考虑的领域大小
        # ksizesobel求导中使用的窗口大小
        # k是harris角点检测方程中的自由参数，取值范围[0.04, 0.06]
        # 这里的膨胀操作使标记的点膨胀，当然没有这步也可以。
        dst = cv2.dilate(dst, None)
        # 保留角点
        gray[dst > 0.01 * dst.max()] = 255
        temp = []
        for line in list(gray):
            temp += list(line)
        line_str = ""
        for point in temp:
            line_str += str(point) + ","
        line_str += str(index) + "\n"
        with open("train.csv", mode="a") as f:
            f.write(line_str)
