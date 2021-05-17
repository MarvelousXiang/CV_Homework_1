import joblib

print("*"*100)
print("Loading val data, please waiting")
print("*"*100)
with open("val.csv", mode="r") as f:
    raw_lines = f.readlines()
    f.close()
str_lines = []
for raw_line in raw_lines:
    temp_line = raw_line.strip("\n").split(",")
    str_lines.append(temp_line)
file_name = []
data = []
for line in str_lines:
    file_name.append(line[-1])
    del line[-1]
    temp = []
    for item in line:
        temp.append(int(item)/255)
    data.append(temp)
print("Loading model, please waiting")
print("*" * 100)
model = joblib.load("img_recognize.pkl")
result = model.predict(data)
print("Writing results, please waiting")
print("*"*100)
with open("val_predict.txt", mode="w") as f:
    for i in range(len(result)):
        f.write(file_name[i] + " " + str(result[i]) + "\n")
    f.close()

