import numpy as np

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