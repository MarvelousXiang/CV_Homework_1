import joblib
from sklearn.neighbors import KNeighborsClassifier
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

print('start training!')
clf = KNeighborsClassifier(n_neighbors=100)
clf.fit(data, target)
print('end training!')
joblib.dump(clf, "img_recognize_knn.pkl")
