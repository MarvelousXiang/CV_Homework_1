from sklearn.neural_network import MLPClassifier
import joblib

print("*" * 100)
print("Training data are been loading now! Please wait ")
print("*" * 100)
data = []
target = []
with open("train.csv", mode="r") as f:
    lines = f.readlines()
    f.close()
for line in lines:
    temp_line = line.strip("\n").split(",")
    number_line = []
    for item in temp_line:
        number_line.append(int(item)/255)
    data.append(number_line[0:len(number_line) - 1])
    target.append(int(number_line[-1]*255))
print("Training data have been loaded successfully, now training starts!")
print("*" * 100)
model = MLPClassifier([4096+2048, 2048], learning_rate_init=0.001, activation='relu', \
                      solver='adam', alpha=0.0001, max_iter=20000)  # 神经网络
print('start training!')
model.fit(data, target)
print('end training!')
joblib.dump(model, "img_recognize.pkl")

