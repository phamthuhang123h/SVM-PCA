import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score #thêm vào để tính chỉ số của mô hình
from sklearn.metrics import recall_score  #GT THU HOI
from sklearn.metrics import f1_score #GT TB GIỮA GT CHÍNH XÁC VÀ GT THU HỒI
from sklearn.tree import DecisionTreeClassifier
from tkinter import * #thêm vào thư viện để tạo form
from tkinter import messagebox
from tkinter import ttk
def tyledung(y_test, y_pred):
    d = 0
    for i in range(len(y_pred)): #đánh giá tỉ lệ mẫu
        if (y_pred[i] == y_test[i]): 
            d = d + 1
    rate = d / len(y_pred)  # tỉ lệ % dự đoán đúng
    return rate
#doc du lieu tu file

data = pd.read_csv('drug.csv') 
le = LabelEncoder() #ham chuyen chuoi thanh so
data['Age'] = le.fit_transform(data['Age']) 
data['Sex'] = le.fit_transform(data['Sex']) 
data['BP'] = le.fit_transform(data['BP'])
data['Cholesterol'] = le.fit_transform(data['Cholesterol'])
data['Na_to_K'] = le.fit_transform(data['Na_to_K'])
X_data = np.array(data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values)
y = np.array(data['Drug'])




max_svm = 0
max_id3 = 0
for j in range(1,6):
    print("lan", j)
    pca = decomposition.PCA(n_components=j) #với mỗi j thì dùng PCA để chọn ra thành phần chính tốt nhất
    pca.fit(X_data) #sử dụng thành phần chính tốt nhất để huấn luyện mô hình

    Xbar = pca.transform(X_data)  # Áp dụng giảm kích thước cho X. #dùng hàm transform trên X để chuyển X ban đầu thành X 1 chiều tốt nhất
    X_train, X_test, y_train, y_test = train_test_split(Xbar, y, test_size=0.3, shuffle=False)#chia dữ liệu thành tập train-test

    id3 = DecisionTreeClassifier(criterion='entropy')# sử dụng mô hình ID3 để huấn luyện trên tập dữ liệu có 1 thành phần chính tốt nhất
    id3.fit(X_train, y_train) #Truyền X train,y train vào mô hình id3
    svc = SVC(kernel='linear')
    svc.fit(X_train, y_train) #Truyền X train,y train vào mô hình svm
    
    y_pred_svm= svc.predict(X_test) #sd mô hình svm để dự đoán X test
    rate = tyledung(y_test, y_pred_svm)
    print('Ty le du doan dung svm: ', rate)
    y_pred_id3 = id3.predict(X_test) #sd mô hình id3 để dự đoán X test
    rate1 = tyledung(y_test,y_pred_id3)
    print('Ty le du doan dung id3: ', rate1)
    if (rate > max_svm): # số mẫu gán là đúng ở thời điểm hiện tại > max thì mô hình hiện tại là moo hình tốt
        num_pca_svm = j
        pca_best = pca #lưu lại PCA tốt nhất
        max_svm = rate
        modelmax_svm = svc #lưu lại mô hình
    if (rate1 > max_id3): # số mẫu gán là đúng ở thời điểm hiện tại > max thì mô hình hiện tại là moo hình tốt
        num_pca_id3 = j
        pca_best = pca #lưu lại PCA tốt nhất
        max_id3 = rate
        modelmax_id3 = id3 #lưu lại mô hình
print("max_svm", max_svm, "d=", num_pca_svm)
print("max_id3", max_id3, "d=", num_pca_id3)


#dt_Train, dt_Test = train_test_split(data, test_size = 0.3, shuffle = False)

#X_train = dt_Train[:, :5]
#y_train = dt_Train[:, 5]
#X_test = dt_Test[:, :5]
#y_test = dt_Test[:, 5]

#svm = SVC(kernel='linear')
#svm.fit(X_train, y_train) #lay du lieu tu tap train de huan luyen mo hinh svm

#---------------------------------------------------------------
#form
form = Tk()
form.title("Dự đoán loại thuốc cho bệnh nhân:")
form.geometry("1000x500")



lable_ten = Label(form, text = "Nhập thông tin cho bệnh nhân:", font=("Arial Bold", 10), fg="red")
lable_ten.grid(row = 1, column = 1, padx = 40, pady = 10)

lable_age = Label(form, text = " Age:")
lable_age.grid(row = 2, column = 1, padx = 40, pady = 10)
textbox_age = Entry(form)
textbox_age.grid(row = 2, column = 2)

lable_sex = Label(form, text = "Sex:")
lable_sex.grid(row = 3, column = 1, pady = 10)
textbox_sex = Entry(form)
textbox_sex.grid(row = 3, column = 2)

lable_bp = Label(form, text = "BP:")
lable_bp.grid(row = 4, column = 1,pady = 10)
textbox_bp = Entry(form)
textbox_bp.grid(row = 4, column = 2)

lable_cholesterol = Label(form, text = "Cholesterol:")
lable_cholesterol.grid(row = 5, column = 1, pady = 10)
textbox_cholesterol = Entry(form)
textbox_cholesterol.grid(row = 5, column = 2)

lable_na_to_k = Label(form, text = "Na_to_K:")
lable_na_to_k.grid(row = 6, column = 1, pady = 10 )
textbox_na_to_k = Entry(form)
textbox_na_to_k.grid(row = 6, column = 2)

#svm

def khanangsvm():
    y_predsvm = svc.predict(X_test)
    dem=0
    for i in range (len(y_predsvm)):
        if(y_predsvm[i] == y_test[i]):
            dem= dem+1
    count = (dem/len(y_predsvm))*100
    lbl1.configure(text= count)
    return count
#button_svm1 = Button(form, text = 'Khả năng dự đoán đúng ', command = khanangsvm)
#button_svm1.grid(row = 10, column = 1, padx = 30)
#lbl1 = Label(form, text="...")
#lbl1.grid(column=2, row=10)

y_predsvm = svc.predict(X_test)
lbl1 = Label(form)
lbl1.grid(column=1, row=8)
lbl1.configure(text="Tỉ lệ dự đoán đúng của SVM: "+'\n'
                           +"Precision: "+str(precision_score(y_test, y_predsvm, average='macro')*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test, y_predsvm, average='macro')*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test, y_predsvm, average='macro')*100)+"%"+'\n'
                           +"Accuracy: " +str(khanangsvm()))

def chuyen_doi_encoder(data):
    for i in range(5):
        if data[i]=="F":
            data[i]=0
        elif (data[i] == "M"):
            data[i] = 1
        elif (data[i] == "LOW"):
            data[i] = 2
        elif (data[i] == "NORMAL"):
            data[i] = 3
        elif (data[i] == "HIGH"):
            data[i] = 4
    return data
def dudoansvm():
    age = textbox_age.get()
    sex = textbox_sex.get()
    bp = textbox_bp.get()
    cholesterol = textbox_cholesterol.get()
    na_to_k =textbox_na_to_k.get()
    
    if((sex == '') or (age == '') or (bp == '') or (cholesterol == '') or (na_to_k == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = chuyen_doi_encoder(np.array([age,sex,bp,cholesterol,na_to_k])).reshape(1, -1)
        #sample_svm=pca.transform(X_dudoan)
        y_kqua = modelmax_svm.predict(X_dudoan)
        lbl.configure(text= y_kqua)
        print(X_dudoan)
button_svm = Button(form, text = 'Kết quả dự đoán theo SVM', command = dudoansvm)
button_svm.grid(row = 9, column = 1, pady = 20)
lbl = Label(form, text='...')
lbl.grid(column=2, row=9)





#Cay quyet dinh
#dudoanid3test
def khanangid3():
    y_id3 = id3.predict(X_test)
    dem=0
    for i in range (len(y_id3)):
        if(y_id3[i] == y_test[i]):
            dem= dem+1
    count = (dem/len(y_test))*100
    lbl3.configure(text= count)
    return count

y_id3 = id3.predict(X_test)
lbl3 = Label(form)
lbl3.grid(column=3, row=8)
lbl3.configure(text="Tỉ lệ dự đoán đúng của ID3: "+'\n'
                           +"Precision: "+str(precision_score(y_test, y_id3, average='macro')*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test, y_id3, average='macro')*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test, y_id3, average='macro')*100)+"%"+'\n'
                           +"Accuracy: "+str(khanangid3()))
def dudoanid3():
    age = textbox_age.get()
    sex = textbox_sex.get()
    bp = textbox_bp.get()
    cholesterol = textbox_cholesterol.get()
    na_to_k =textbox_na_to_k.get()
    
    if((sex == '') or (age == '') or (bp == '') or (cholesterol == '') or (na_to_k == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = chuyen_doi_encoder(np.array([age,sex,bp,cholesterol,na_to_k])).reshape(1, -1)
        y_kqua = modelmax_id3.predict(X_dudoan)
        lbl2.configure(text= y_kqua)
        
button_id3 = Button(form, text = 'Kết quả dự đoán theo ID3', command = dudoanid3)
button_id3.grid(row = 9, column = 3, pady = 20)
lbl2 = Label(form, text="...")
lbl2.grid(column=4, row=9)


#button_id31 = Button(form, text = 'Khả năng dự đoán đúng ', command = khanangid3)
#button_id31.grid(row = 10, column = 3, padx = 30)
#lbl3 = Label(form, text="...")
#lbl3.grid(column=4, row=10)


form.mainloop()
