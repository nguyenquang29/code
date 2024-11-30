import cv2
import numpy as np
from pypylon import pylon

# Tạo đối tượng camera Basler
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Mở camera Basler
camera.Open()

# Bắt đầu quá trình lấy hình ảnh từ camera
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

def nothing(x):
    pass

# Tạo cửa sổ và các thanh trượt (trackbars) để điều chỉnh độ sáng (brightness)
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-Brightness", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U-Brightness", "Trackbars", 255, 255, nothing)

while True:
    # Lấy một khung hình từ camera Basler
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grab_result.GrabSucceeded():
        # Chuyển đổi hình ảnh từ pylon image sang numpy array
        frame = grab_result.Array

        # Kiểm tra nếu hình ảnh có 1 kênh (grayscale)
        if len(frame.shape) == 2:  # Nếu ảnh chỉ có 1 kênh (grayscale)
            # Không cần chuyển đổi sang BGR, xử lý trực tiếp grayscale
            gray = frame
        else:
            # Nếu hình ảnh có 3 kênh (màu), chuyển sang grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Lấy giá trị từ trackbars để điều chỉnh độ sáng (brightness)
        l_brightness = cv2.getTrackbarPos("L-Brightness", "Trackbars")
        u_brightness = cv2.getTrackbarPos("U-Brightness", "Trackbars")

        # Tạo mặt nạ (mask) dựa trên độ sáng (brightness)
        mask = cv2.inRange(gray, l_brightness, u_brightness)

        # Hiển thị khung hình và mặt nạ
        cv2.imshow("Frame", frame)  # Hiển thị ảnh gốc
        cv2.imshow("Gray", gray)    # Hiển thị ảnh grayscale
        cv2.imshow("Mask", mask)    # Hiển thị mặt nạ theo độ sáng
       

    # Kiểm tra nếu người dùng nhấn phím ESC để thoát
    key = cv2.waitKey(1)
    if key == 27:
        break

# Dừng và đóng kết nối camera
camera.StopGrabbing()
camera.Close()

# Đóng tất cả cửa sổ OpenCV
cv2.destroyAllWindows()
