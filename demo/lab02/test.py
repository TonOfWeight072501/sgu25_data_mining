import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from IPython.display import display 
# 1. Tải file .npz
data = np.load(r"C:\Users\kimho\OneDrive\Desktop\Project\Data Mining\GIt_pull\demo\lab02\exps\diabetes\data.npz")

# 2. Truy cập vào từng mảng dữ liệu bằng tên của chúng
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

# 3. Kiểm tra và sử dụng dữ liệu
# In ra hình dạng (shape) của mỗi mảng để xác nhận
print(f"Hình dạng của X_train: {X_train.shape}")
print(f"Hình dạng của y_train: {y_train.shape}")
print(f"Hình dạng của X_test: {X_test.shape}")
print(f"Hình dạng của y_test: {y_test.shape}")

# In ra 5 dòng đầu của tập huấn luyện để xem qua
print("\n--- 5 dòng đầu của X_train ---")
print(X_train[:5])

print("\n--- 5 giá trị đầu của y_train ---")
print(y_train[:5])



# 2. Huấn luyện một mô hình máy học (ví dụ: Hồi quy Logistic)
# Mô hình sẽ học từ 576 mẫu trong tập huấn luyện
model = LogisticRegression(max_iter=1000) # Tăng max_iter để đảm bảo mô hình hội tụ
model.fit(X_train, y_train)

# 3. Tạo ra các dự đoán trên tập kiểm tra
# Mô hình sẽ dự đoán kết quả cho 192 mẫu trong tập kiểm tra
y_pred = model.predict(X_test)

# 4. Tạo một DataFrame để so sánh và phân tích kết quả
# Lấy tên các cột đặc trưng (giả sử theo thứ tự chuẩn)
feature_names = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree', 'age']

# Tạo DataFrame từ dữ liệu kiểm tra
df_results = pd.DataFrame(X_test, columns=feature_names)
df_results['Actual_Class'] = y_test
df_results['Predicted_Class'] = y_pred

# Thêm một cột để xác định dự đoán là Đúng hay Sai
df_results['Status'] = np.where(df_results['Actual_Class'] == df_results['Predicted_Class'], 'Đúng', 'Sai')

# 5. Hiển thị các dự đoán SAI để phân tích lỗi
print("--- CÁC TRƯỜNG HỢP MÔ HÌNH DỰ ĐOÁN SAI ---")
display(df_results[df_results['Status'] == 'Sai'])

print("\n--- CÁC TRƯỜNG HỢP MÔ HÌNH DỰ ĐOÁN ĐÚNG (5 ví dụ) ---")
display(df_results[df_results['Status'] == 'Đúng'].head())