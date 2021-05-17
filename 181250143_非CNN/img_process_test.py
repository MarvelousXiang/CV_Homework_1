import os
import cv2

project_path = os.getcwd()
test_path = project_path + "/test/"

# 读取测试集图片数据
test_dir = os.listdir(test_path)
with open("test.csv", mode="w") as f:
    f.close()
for file in test_dir:
    # 拼接路径便于读取
    img_path = test_path + file
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
    line_str = line_str[0:len(line_str)-1]
    line_str += "\n"
    with open("test.csv", mode="a") as f:
        f.write(line_str)
