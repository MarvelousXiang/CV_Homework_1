'''
from keras.models import load_model
import argparse
import pickle
import cv2


def predict():
    # 加载测试数据并进行相同预处理操作
    image = cv2.imread('./test')
    output = image.copy()
    image = cv2.resize(image, (64, 64))

    # scale图像数据
    image = image.astype("float") / 255.0

    # 对图像进行拉平操作
    image = image.reshape((1, image.shape[0], image.shape[1],image.shape[2]))

    # 读取模型和标签
    print("------读取模型和标签------")
    model = load_model('./output/cnn.model')
    lb = pickle.loads(open('./output/cnn_lb.pickle', "rb").read())

    # 预测
    preds = model.predict(image)

    # 得到预测结果以及其对应的标签
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]

    file = open(r'181250164.txt', mode='w')
    for i in 9999:
        file.write(str(i))
        file.write(".jpg")
        file.write(" ")
        file.write(label)
        file.write('\n')
        file.close()
'''


def load_res(res_path):
    with open(res_path, 'r') as f:
        items = f.readlines()
        items = [item.strip().split() for item in items]
    iid_to_cid = {item[0]: item[1] for item in items}
    return iid_to_cid


def cal_acc(anno, pred):
    sample_num = len(anno)
    hit_cnt = 0.0
    for iid, cid in anno.items():
        if iid in pred and cid == pred[iid]:
            hit_cnt += 1
    return hit_cnt / sample_num


anno_path = 'val_anno.txt'
pred_path = 'val.txt'
anno = load_res(anno_path)
pred = load_res(pred_path)
acc = cal_acc(anno, pred)
print('accuracy: %.4f' % acc)
