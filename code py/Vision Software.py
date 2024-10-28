import sys
import cv2
import numpy as np
import snap7
from snap7.util import get_bool, get_real, set_bool
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QMessageBox, QLineEdit
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QPoint, QRect, QMutex, QWaitCondition
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from ScaleSYS import Ui_MainWindow
import time
import json
import os
import asyncio
from snap7.types import Areas
from snap7.util import set_int
from snap7.types import Areas
import sys
import cv2
import numpy as np
import snap7
from snap7.util import get_bool, get_real, set_bool
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QMessageBox, QLineEdit
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QPoint, QRect, QMutex, QWaitCondition
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from ScaleSYS import Ui_MainWindow
import time
import json
import os
import asyncio
from snap7.types import Areas
from snap7.util import set_int, get_real
from snap7.types import Areas

plc = snap7.client.Client()
IP = "192.168.0.1"
DB_NUMBER = 1
START_ADDRESS = 0
class VideoCaptureThread(QThread):
        change_pixmap_signal = pyqtSignal(QImage)
        update_coords_signal = pyqtSignal(int, int)
        update_object_count_signal = pyqtSignal(int)  # New signal for object count
        update_plc_data_signal = pyqtSignal(float, float, float,float,float,float,float,float)  # New signal for PLC data
        def __init__(self, plc, parent=None):
            super().__init__(parent)
            self._run_flag = True
            self.plc = plc
            self.roi_x = 0.25
            self.roi_y = 0.25
            self.roi_width = 0.5
            self.roi_height = 0.5
            self.contour_threshold = 500  # Default value
            self.brightness = 0  # New attribute for brightness
            self.lower_red = np.array([0, 53, 90])  # Thêm thuộc tính lower_red
            self.upper_red = np.array([142, 154, 228])
        def set_lower_red(self, lower_red):
            self.lower_red = np.array(lower_red)  # Chuyển đổi list thành numpy array
        def set_upper_red(self, upper_red):
            self.upper_red = np.array(upper_red)
        def set_roi(self, x, y, width, height):
            self.roi_x = x
            self.roi_y = y
            self.roi_width = width
            self.roi_height = height

        def set_contour_threshold(self, value):
            self.contour_threshold = value
           
        def set_brightness(self, value):  # New method to set brightness
            self.brightness = value

        def stop(self):
            self._run_flag = False
            self.wait()  # Wait for the thread to finish gracefully
            cv2.destroyAllWindows()
        def run(self):
            cap = cv2.VideoCapture(0)
            fps = 15  #15 hoặc 30
            delay = 1 / fps
            while self._run_flag:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (640, 480)) 
                    frame = cv2.convertScaleAbs(frame, alpha=1, beta=self.brightness)
                    height, width, _ = frame.shape
                    roi_top_left_x = int(width * self.roi_x)
                    roi_top_left_y = int(height * self.roi_y)
                    roi_bottom_right_x = int(width * (self.roi_x + self.roi_width))
                    roi_bottom_right_y = int(height * (self.roi_y + self.roi_height))

                    cv2.rectangle(frame, 
                                (roi_top_left_x, roi_top_left_y), 
                                (roi_bottom_right_x, roi_bottom_right_y), 
                                (0, 255, 0), 2)
                    
                    roi_frame = frame[roi_top_left_y:roi_bottom_right_y, roi_top_left_x:roi_bottom_right_x]

                    hsv_image = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

                    mask1 = cv2.inRange(hsv_image, self.lower_red, self.upper_red)

                    contours, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    detected_objects = []
                   
                    for contour in contours:
                        if cv2.contourArea(contour) > self.contour_threshold:
                            x, y, w, h = cv2.boundingRect(contour)
                            cX = x + w // 2 + roi_top_left_x
                            cY = y + h // 2 + roi_top_left_y

                            detected_objects.append((cX, cY))

                            cv2.rectangle(frame, (x + roi_top_left_x, y + roi_top_left_y), 
                                        (x + roi_top_left_x + w, y + roi_top_left_y + h), (0, 255, 0), 2)
                            cv2.circle(frame, (cX, cY), 7, (255, 0, 0), -1)
                            cv2.putText(frame, f"X: {cX} Y: {cY}", (cX - 20, cY - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    # Emit the number of detected objects
                    self.update_object_count_signal.emit(len(detected_objects))
                    # Read PLC data
                    if self.plc.get_connected():
                        try:
                            delta_x = get_real(self.plc.db_read(1, 52, 4),0)
                            delta_y = get_real(self.plc.db_read(1, 56, 4),0)
                            delta_z = get_real(self.plc.db_read(1, 60, 4),0)
                            axis_x = get_real(self.plc.db_read(1, 64, 4),0)
                            axis_y = get_real(self.plc.db_read(1, 68, 4),0)
                            axis_z = get_real(self.plc.db_read(1, 72, 4),0)
                            tbTocdoBT = get_real(self.plc.db_read(1, 76, 4),0)
                            tbTocdoRB = get_real(self.plc.db_read(1, 80, 4),0)
                            self.update_plc_data_signal.emit(delta_x, delta_y, delta_z,axis_x,axis_y,axis_z,tbTocdoBT,tbTocdoRB)
                        except Exception as e:
                            print(f"Error reading PLC data: {e}")
                    detected_objects.sort(key=lambda obj: (obj[1], obj[0]))
                    if detected_objects:
                        cX, cY = detected_objects[0]
                        self.update_coords_signal.emit(cX, cY)

                        max_retries = 3
                        retries = 0

                        while retries < max_retries:
                            try:
                                if self.plc.get_connected():
                                    reading = bytearray(4)
                                    snap7.util.set_int(reading, 0, cX)
                                    snap7.util.set_int(reading, 2, cY)
                                    self.plc.db_write(1, 8, reading)
                                    
                                    boolean_value = bytearray(1)
                                    boolean_value[0] = 1
                                    self.plc.db_write(1, 18, boolean_value)
                                    break  # Exit loop if successful
                                else:
                                    print("PLC is not connected")
                                    break  # Exit loop if PLC is not connected
                            except Exception as e:
                                print(f"Error writing coordinates to PLC (Retry {retries + 1}/{max_retries}): {e}")
                                retries += 1
                                time.sleep(1)  # Wait before retrying
                    else:
                        max_retries = 3
                        retries = 0

                        while retries < max_retries:
                            try:
                                if self.plc.get_connected():
                                    reading = bytearray(4)
                                    snap7.util.set_int(reading, 0, 0)
                                    snap7.util.set_int(reading, 2, 0)
                                    self.plc.db_write(1, 8, reading)
                                    
                                    boolean_value = bytearray(1)
                                    boolean_value[0] = 0
                                    self.plc.db_write(1, 18, boolean_value)
                                    break  # Exit loop if successful
                                else:
                                    print("PLC is not connected")
                                    break  # Exit loop if PLC is not connected
                            except Exception as e:
                                print(f"Error writing to PLC (Retry {retries + 1}/{max_retries}): {e}")
                                retries += 1
                                time.sleep(1)  # Wait before retrying
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.change_pixmap_signal.emit(qt_image)
                time.sleep(delay)    
            cap.release()
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)

        self.plc = snap7.client.Client()
        try:
            self.plc.connect('192.168.0.1', 0, 1)
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", f"Could not connect to PLC: {str(e)}")
            sys.exit(1)

        # Initialize attributes related to image processing
        self.roi_x = 0.25
        self.roi_y = 0.25
        self.roi_width = 0.5
        self.roi_height = 0.5
        self.contour_threshold = 500
        self.brightness = 0
         # Khởi tạo giá trị mặc định cho lower_red
        self.lower_red = [0, 53, 90]
        self.upper_red = [142, 154, 228]
         # Cập nhật giá trị lên lineEdit
        self.update_lower_red_values()
        self.update_upper_red_values()
        # Kết nối sự kiện textChanged của các lineEdit với phương thức cập nhật
        self.uic.lineEdit.textChanged.connect(self.update_lower_red_from_ui)
        self.uic.lineEdit_2.textChanged.connect(self.update_lower_red_from_ui)
        self.uic.lineEdit_3.textChanged.connect(self.update_lower_red_from_ui)
        # Kết nối sự kiện textChanged của các lineEdit với phương thức cập nhật
        self.uic.lineEdit_6.textChanged.connect(self.update_upper_red_from_ui)
        self.uic.lineEdit_4.textChanged.connect(self.update_upper_red_from_ui)
        self.uic.lineEdit_5.textChanged.connect(self.update_upper_red_from_ui)
        # Load settings and update UI components
       
        self.uic.SLDetect.setValue(self.contour_threshold)
        self.uic.Slider_2.setValue(self.brightness)   
        # Add initialization for brightness slider
        self.brightness = 0
        self.uic.Slider_2.setMinimum(-100)
        self.uic.Slider_2.setMaximum(100)
        self.uic.Slider_2.setValue(self.brightness)
        self.uic.Slider_2.valueChanged.connect(self.update_brightness)

        

        self.plc_mutex = QMutex()
        self.plc_wait_condition = QWaitCondition()
        self.last_image = None
        self.selection_rect = QRect()
    
        self.uic.lbImage.setFixedSize(400, 500)
        self.uic.lbImage.setAlignment(Qt.AlignCenter)
        
        self.uic.btHome.clicked.connect(self.Home_Mode)
        self.uic.btStart.clicked.connect(self.Start_Mode)
        self.uic.btReset.clicked.connect(self.Reset_Mode)
        self.uic.btStop.clicked.connect(self.Stop_Mode)
        self.uic.btVaccum.clicked.connect(self.Vacuum)
        self.uic.btBangtai1.clicked.connect(self.BT1)
        self.uic.btBangtai2.clicked.connect(self.BT2)


        self.thread = VideoCaptureThread(self.plc)
        self.thread.update_object_count_signal.connect(self.update_object_count)  # Connect new signal
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_coords_signal.connect(self.update_coords)
        self.thread.update_plc_data_signal.connect(self.update_plc_data)  # Connect new signal

        self.uic.btConnect.clicked.connect(self.start_webcam)
        self.uic.btDisconnect.clicked.connect(self.stop_webcam)
        
        self.uic.btXPlus.clicked.connect(self.adjust_roi_x_plus)
        self.uic.btXMinus.clicked.connect(self.adjust_roi_x_minus)
        self.uic.btYPlus.clicked.connect(self.adjust_roi_y_plus)
        self.uic.btYMinus.clicked.connect(self.adjust_roi_y_minus)
        self.uic.btWidthPlus.clicked.connect(self.adjust_roi_width_plus)
        self.uic.btWidthMinus.clicked.connect(self.adjust_roi_width_minus)
        self.uic.btHeightPlus.clicked.connect(self.adjust_roi_height_plus)
        self.uic.btHeightMinus.clicked.connect(self.adjust_roi_height_minus)

        self.uic.doubleSpinBox.valueChanged.connect(self.update_plc_db_byte0)
        self.uic.doubleSpinBox_2.valueChanged.connect(self.update_plc_db_byte2)

        
        # Add initialization for contour threshold
        self.contour_threshold = 500  # Default value
        self.uic.SLDetect.setMinimum(100)
        self.uic.SLDetect.setMaximum(1000)
        self.uic.SLDetect.setValue(self.contour_threshold)
        self.uic.SLDetect.valueChanged.connect(self.update_contour_threshold)

        
        self.load_settings()
         
    def update_object_count(self, count):
        # Update the lSoluong textbox with the number of detected objects
        self.uic.lSoluong.setText(str(count))
    def update_plc_data(self, delta_x, delta_y, delta_z,axis_x,axis_y,axis_z,tbTocdoBT,tbTocdoRB):
        self.uic.tbThetaX.setText(f"{delta_x:.2f}")
        self.uic.tbThetaY.setText(f"{delta_y:.2f}")
        self.uic.tbThetaZ.setText(f"{delta_z:.2f}")
        self.uic.tbAxisX.setText(f"{axis_x:.2f}")
        self.uic.tbAxisY.setText(f"{axis_y:.2f}")
        self.uic.tbAxisZ.setText(f"{axis_z:.2f}")
        self.uic.lTocdoBT.setText(f"{tbTocdoBT:.2f}")
        self.uic.lTocdoRB.setText(f"{tbTocdoRB:.2f}")        
    def update_contour_threshold(self, value):
        self.contour_threshold = value
        self.thread.set_contour_threshold(value)
        value = self.contour_threshold
        # Hiển thị giá trị lên label numPH
        self.uic.numPH.setText(str(value))
        self.save_settings()

    def update_brightness(self, value):
        self.brightness = value
        self.thread.set_brightness(value)
        value = self.brightness
        # Hiển thị giá trị lên label numPH
        self.uic.numDS.setText(str(value))
        self.save_settings()
    def update_lower_red_values(self):
        # Cập nhật giá trị từ lower_red lên lineEdit
        self.uic.lineEdit.setText(str(self.lower_red[0]))
        self.uic.lineEdit_2.setText(str(self.lower_red[1]))
        self.uic.lineEdit_3.setText(str(self.lower_red[2]))
    
        
    def update_lower_red_from_ui(self):
        # Cập nhật giá trị lower_red từ lineEdit
        try:
            self.lower_red[0] = int(self.uic.lineEdit.text())
            self.lower_red[1] = int(self.uic.lineEdit_2.text())
            self.lower_red[2] = int(self.uic.lineEdit_3.text())
            
            # Cập nhật giá trị trong VideoCaptureThread
            if hasattr(self, 'thread') and self.thread is not None:
                self.thread.set_lower_red(self.lower_red)
            
            # Lưu cài đặt sau khi cập nhật
            self.save_settings()
        except ValueError:
            # Xử lý trường hợp người dùng nhập giá trị không hợp lệ
            print("Vui lòng nhập giá trị số nguyên hợp lệ.")
    def update_upper_red_values(self):
        self.uic.lineEdit_6.setText(str(self.upper_red[0]))
        self.uic.lineEdit_4.setText(str(self.upper_red[1]))
        self.uic.lineEdit_5.setText(str(self.upper_red[2]))

    def update_upper_red_from_ui(self):
        try:
            self.upper_red[0] = int(self.uic.lineEdit_6.text())
            self.upper_red[1] = int(self.uic.lineEdit_4.text())
            self.upper_red[2] = int(self.uic.lineEdit_5.text())
            
            if hasattr(self, 'thread') and self.thread is not None:
                self.thread.set_upper_red(self.upper_red)
            
            self.save_settings()
        except ValueError:
            print("Please enter valid integer values for upper red.")
            self.spinbox_connections = [
                (self.uic.doubleSpinBox, self.update_plc_db_byte0),
                (self.uic.doubleSpinBox_2, self.update_plc_db_byte2)
        ]
    def save_settings(self):
        settings_data = {
            'roi_x': self.roi_x,
            'roi_y': self.roi_y,
            'roi_width': self.roi_width,
            'roi_height': self.roi_height,
            'contour_threshold': self.contour_threshold,
            'brightness': self.brightness,
            'doubleSpinBox_value': self.uic.doubleSpinBox.value(),
            'doubleSpinBox_2_value': self.uic.doubleSpinBox_2.value(),
            'lower_red': self.lower_red,
            'upper_red': self.upper_red
        }
        with open('settings.txt', 'w') as f:
            json.dump(settings_data, f)

    def load_settings(self):
        if os.path.exists('settings.txt'):
            with open('settings.txt', 'r') as f:
                settings_data = json.load(f)
            self.roi_x = settings_data.get('roi_x', self.roi_x)
            self.roi_y = settings_data.get('roi_y', self.roi_y)
            self.roi_width = settings_data.get('roi_width', self.roi_width)
            self.roi_height = settings_data.get('roi_height', self.roi_height)
            self.contour_threshold = settings_data.get('contour_threshold', self.contour_threshold)
            self.brightness = settings_data.get('brightness', self.brightness)
            self.uic.doubleSpinBox.setValue(settings_data.get('doubleSpinBox_value', 0.0))
            self.uic.doubleSpinBox_2.setValue(settings_data.get('doubleSpinBox_2_value', 0.0))
            self.uic.SLDetect.setValue(self.contour_threshold)
            self.uic.Slider_2.setValue(self.brightness)
            
            # Tải giá trị lower_red từ settings
            self.lower_red = settings_data.get('lower_red', self.lower_red)
            self.upper_red = settings_data.get('upper_red', self.upper_red)
            
            # Cập nhật giá trị trong thread nếu thread đã được khởi tạo
            if hasattr(self, 'thread') and self.thread is not None:
                self.thread.set_roi(self.roi_x, self.roi_y, self.roi_width, self.roi_height)
                self.thread.set_contour_threshold(self.contour_threshold)
                self.thread.set_brightness(self.brightness)
                self.thread.set_lower_red(self.lower_red)
                self.thread.set_upper_red(self.upper_red)
        # Cập nhật giao diện người dùng sau khi tải settings
        self.update_lower_red_values()
        self.update_upper_red_values()

            
            
    
    def adjust_roi_x_plus(self):
        self.roi_x = min(self.roi_x + 0.01, 1 - self.roi_width)
        self.update_roi()

    def adjust_roi_x_minus(self):
        self.roi_x = max(self.roi_x - 0.01, 0)
        self.update_roi()

    def adjust_roi_y_plus(self):
        self.roi_y = min(self.roi_y + 0.01, 1 - self.roi_height)
        self.update_roi()

    def adjust_roi_y_minus(self):
        self.roi_y = max(self.roi_y - 0.01, 0)
        self.update_roi()

    def adjust_roi_width_plus(self):
        self.roi_width = min(self.roi_width + 0.01, 1 - self.roi_x)
        self.update_roi()

    def adjust_roi_width_minus(self):
        self.roi_width = max(self.roi_width - 0.01, 0.1)
        self.update_roi()

    def adjust_roi_height_plus(self):
        self.roi_height = min(self.roi_height + 0.01, 1 - self.roi_y)
        self.update_roi()

    def adjust_roi_height_minus(self):
        self.roi_height = max(self.roi_height - 0.01, 0.1)
        self.update_roi()

    def update_roi(self):
        self.thread.set_roi(self.roi_x, self.roi_y, self.roi_width, self.roi_height)
        self.save_settings()
    def update_plc_db_byte0(self, value):
        try:
            if self.plc.get_connected():
                reading = bytearray(2) # Use bytearray of length 2 for int (16-bit)
                snap7.util.set_int(reading, 0, int(value)) # Convert to int and write
                self.plc.db_write(1, 0, reading) 
            else:
                print("PLC is not connected")
        except Exception as e:
            print(f"Error writing to PLC DB byte 0: {e}")

    def update_plc_db_byte2(self, value):
        try:
            if self.plc.get_connected():
                reading = bytearray(2) # Use bytearray of length 2 for int (16-bit)
                snap7.util.set_int(reading, 0, int(value)) # Convert to int and write
                self.plc.db_write(1, 2, reading)
            else:
                print("PLC is not connected")
        except Exception as e:
            print(f"Error writing to PLC DB byte 2: {e}")
    def Start_Mode(self):
        try:
            if self.plc.get_connected():
                # Đọc dữ liệu hiện tại từ DB1 tại byte offset 18
                db_data = self.plc.db_read(1, 18, 1)  # Đọc 1 byte từ DB1, bắt đầu từ byte offset 18
                
                # Tạo mảng byte từ dữ liệu đọc được
                current_byte = bytearray(db_data)[0]

                # Kiểm tra giá trị của bit 18.1 (bit 1 trong byte)
                bit_position = 1  # Bit 1 trong byte (bit 18.1)
                is_bit_on = (current_byte & (1 << bit_position)) != 0
                
                if not is_bit_on:
                    # Nếu bit 18.1 là OFF, thực hiện ghi dữ liệu mới
                    new_byte = current_byte | (1 << bit_position)  # Đặt bit 18.1 thành ON
                    self.plc.db_write(1, 18, bytearray([new_byte]))
                    print("Home signal sent.")
                else:
                    print("Bit 18.1 is already ON. No action taken.")
                    
                # Thêm độ trễ để đảm bảo PLC có đủ thời gian xử lý yêu cầu
                time.sleep(0)  

            else:
                print("PLC is not connected")
        except Exception as e:
            print(f"Error sending home signal: {e}")

    def Home_Mode(self):
        try:
            if self.plc.get_connected():
                # Đọc dữ liệu hiện tại từ DB1 tại byte offset 18
                db_data = self.plc.db_read(1, 18, 1)  # Đọc 1 byte từ DB1, bắt đầu từ byte offset 18
                
                # Tạo mảng byte từ dữ liệu đọc được
                current_byte = bytearray(db_data)[0]

                # Kiểm tra giá trị của bit 18.2 (bit 2 trong byte)
                bit_position = 4  # Bit 2 trong byte (bit 18.2)
                is_bit_on = (current_byte & (1 << bit_position)) != 0
                
                if not is_bit_on:
                    # Nếu bit 18.2 là OFF, thực hiện ghi dữ liệu mới
                    new_byte = current_byte | (1 << bit_position)  # Đặt bit 18.2 thành ON
                    self.plc.db_write(1, 18, bytearray([new_byte]))
                    print("Start signal sent.")
                else:
                    print("Bit 18.2 is already ON. No action taken.")
                    
                # Thêm độ trễ để đảm bảo PLC có đủ thời gian xử lý yêu cầu
                time.sleep(0)  

            else:
                print("PLC is not connected")
        except Exception as e:
            print(f"Error sending start signal: {e}")

    def Stop_Mode(self):
        try:
            if self.plc.get_connected():
                # Đọc dữ liệu hiện tại từ DB1 tại byte offset 18
                db_data = self.plc.db_read(1, 18, 1)  # Đọc 1 byte từ DB1, bắt đầu từ byte offset 18
                
                # Tạo mảng byte từ dữ liệu đọc được
                current_byte = bytearray(db_data)[0]

                # Kiểm tra giá trị của bit 18.2 (bit 2 trong byte)
                bit_position = 2  # Bit 2 trong byte (bit 18.2)
                is_bit_on = (current_byte & (1 << bit_position)) != 0
                
                if not is_bit_on:
                    # Nếu bit 18.2 là OFF, thực hiện ghi dữ liệu mới
                    new_byte = current_byte | (1 << bit_position)  # Đặt bit 18.2 thành ON
                    self.plc.db_write(1, 18, bytearray([new_byte]))
                    print("Start signal sent.")
                else:
                    print("Bit 18.2 is already ON. No action taken.")
                    
                # Thêm độ trễ để đảm bảo PLC có đủ thời gian xử lý yêu cầu
                time.sleep(0)  

            else:
                print("PLC is not connected")
        except Exception as e:
            print(f"Error sending start signal: {e}")

    def Reset_Mode(self):
        try:
            if self.plc.get_connected():
                # Đọc dữ liệu hiện tại từ DB1 tại byte offset 18
                db_data = self.plc.db_read(1, 18, 1)  # Đọc 1 byte từ DB1, bắt đầu từ byte offset 18
                
                # Tạo mảng byte từ dữ liệu đọc được
                current_byte = bytearray(db_data)[0]

                # Kiểm tra giá trị của bit 18.2 (bit 2 trong byte)
                bit_position = 3  # Bit 2 trong byte (bit 18.2)
                is_bit_on = (current_byte & (1 << bit_position)) != 0
                
                if not is_bit_on:
                    # Nếu bit 18.2 là OFF, thực hiện ghi dữ liệu mới
                    new_byte = current_byte | (1 << bit_position)  # Đặt bit 18.2 thành ON
                    self.plc.db_write(1, 18, bytearray([new_byte]))
                    print("Start signal sent.")
                else:
                    print("Bit 18.2 is already ON. No action taken.")
                    
                # Thêm độ trễ để đảm bảo PLC có đủ thời gian xử lý yêu cầu
                time.sleep(0)  

            else:
                print("PLC is not connected")
        except Exception as e:
            print(f"Error sending start signal: {e}")
    def Vacuum (self):
        try:
            if self.plc.get_connected():
                # Đọc dữ liệu hiện tại từ DB1 tại byte offset 18
                db_data = self.plc.db_read(1, 18, 1)  # Đọc 1 byte từ DB1, bắt đầu từ byte offset 18
                
                # Tạo mảng byte từ dữ liệu đọc được
                current_byte = bytearray(db_data)[0]

                # Kiểm tra giá trị của bit 18.2 (bit 2 trong byte)
                bit_position = 5  # Bit 2 trong byte (bit 18.2)
                is_bit_on = (current_byte & (1 << bit_position)) != 0
                
                if not is_bit_on:
                    # Nếu bit 18.2 là OFF, thực hiện ghi dữ liệu mới
                    new_byte = current_byte | (1 << bit_position)  # Đặt bit 18.2 thành ON
                    self.plc.db_write(1, 18, bytearray([new_byte]))
                    print("Start signal sent.")
                else:
                    print("Bit 18.2 is already ON. No action taken.")
                    
                # Thêm độ trễ để đảm bảo PLC có đủ thời gian xử lý yêu cầu
                time.sleep(0)  

            else:
                print("PLC is not connected")
        except Exception as e:
            print(f"Error sending start signal: {e}")
    def BT1 (self):
        try:
            if self.plc.get_connected():
                # Đọc dữ liệu hiện tại từ DB1 tại byte offset 18
                db_data = self.plc.db_read(1, 18, 1)  # Đọc 1 byte từ DB1, bắt đầu từ byte offset 18
                
                # Tạo mảng byte từ dữ liệu đọc được
                current_byte = bytearray(db_data)[0]

                # Kiểm tra giá trị của bit 18.2 (bit 2 trong byte)
                bit_position = 6  # Bit 2 trong byte (bit 18.2)
                is_bit_on = (current_byte & (1 << bit_position)) != 0
                
                if not is_bit_on:
                    # Nếu bit 18.2 là OFF, thực hiện ghi dữ liệu mới
                    new_byte = current_byte | (1 << bit_position)  # Đặt bit 18.2 thành ON
                    self.plc.db_write(1, 18, bytearray([new_byte]))
                    print("Start signal sent.")
                else:
                    print("Bit 18.2 is already ON. No action taken.")
                    
                # Thêm độ trễ để đảm bảo PLC có đủ thời gian xử lý yêu cầu
                time.sleep(0)  

            else:
                print("PLC is not connected")
        except Exception as e:
            print(f"Error sending start signal: {e}")
    def BT2 (self):
        try:
            if self.plc.get_connected():
                # Đọc dữ liệu hiện tại từ DB1 tại byte offset 18
                db_data = self.plc.db_read(1, 18, 1)  # Đọc 1 byte từ DB1, bắt đầu từ byte offset 18
                
                # Tạo mảng byte từ dữ liệu đọc được
                current_byte = bytearray(db_data)[0]

                # Kiểm tra giá trị của bit 18.2 (bit 2 trong byte)
                bit_position = 7  # Bit 2 trong byte (bit 18.2)
                is_bit_on = (current_byte & (1 << bit_position)) != 0
                
                if not is_bit_on:
                    # Nếu bit 18.2 là OFF, thực hiện ghi dữ liệu mới
                    new_byte = current_byte | (1 << bit_position)  # Đặt bit 18.2 thành ON
                    self.plc.db_write(1, 18, bytearray([new_byte]))
                    print("Start signal sent.")
                else:
                    print("Bit 18.2 is already ON. No action taken.")
                    
                # Thêm độ trễ để đảm bảo PLC có đủ thời gian xử lý yêu cầu
                time.sleep(0)  

            else:
                print("PLC is not connected")
        except Exception as e:
            print(f"Error sending start signal: {e}")
    def start_webcam(self):
        self.thread.start()
        self.uic.lbStatus.setText("Kết nối thành công")
        self.uic.lbStatus.setStyleSheet("color: white; background-color: green;")
        self.uic.machineStatus.setText("Connected")
        self.uic.machineStatus.setStyleSheet("color: green")
        self.uic.plcStatus.setText("PLC Siemens-1200")
        self.uic.lGreen.setStyleSheet("background-color: green;")


    def stop_webcam(self):
        self.thread.stop()
        # Clear the QLabel
        self.uic.lbImage.clear()
        self.uic.lbStatus.setText("Chưa kết nối")
        self.uic.lbStatus.setStyleSheet("color: black; background-color: none;")
        self.uic.machineStatus.setText("Disonnected")
        self.uic.machineStatus.setStyleSheet("color: red")
        self.uic.plcStatus.setText("____________")
        self.uic.lGreen.setStyleSheet("background-color: red;")

    def update_image(self, qt_image):
        # Thay đổi kích thước hình ảnh thành 400x500
        scaled_image = qt_image.scaled(400, 500, Qt.KeepAspectRatio)
        self.uic.lbImage.setPixmap(QPixmap.fromImage(scaled_image))

    def update_coords(self, x, y):
        # Cập nhật textbox
        self.uic.tbX.setText(str(x))
        self.uic.tbY.setText(str(y))

    def closeEvent(self, event):
        # Save settings
        self.save_settings()
        
        # Stop the webcam thread and wait for it to finish
        self.thread.stop()
        self.thread.wait()
        
        # Disconnect from PLC
        if self.plc.get_connected():
            self.plc.disconnect()
        super().closeEvent(event)
        # Accept the close event
        event.accept()
        super().closeEvent(event)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    try:
        sys.exit(app.exec())
    except SystemExit as e:
        print("Application closed:", e)
