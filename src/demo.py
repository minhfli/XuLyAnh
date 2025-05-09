import tkinter as tk
from tkinter import filedialog, Scale, Label, Button, Frame
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys
import matplotlib.pyplot as plt

# Thêm đường dẫn hiện tại vào sys.path để có thể import module filter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.filter import get_starting_point_v2, cluster_filter_v3, compute_local_mean_var

class EdgePreservingFilterDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Bộ lọc bảo toàn cạnh - Demo")
        self.root.geometry("1200x750")
        
        # Biến lưu trữ hình ảnh và tham số
        self.original_image = None
        self.filtered_image = None
        self.current_image_path = None
        self.alpha = 0.5
        self.k_iterations = 5
        self.kernel_size = 11
        
        # Tạo giao diện
        self.create_ui()
    
    def create_ui(self):
        # Frame chứa điều khiển
        control_frame = Frame(self.root, bg='#f0f0f0')
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Nút tải hình ảnh
        Button(control_frame, text="Tải hình ảnh", command=self.open_image, 
               bg="#4CAF50", fg="white", font=("Arial", 12), padx=10).grid(row=0, column=0, padx=10, pady=10)
        
        # Thanh điều chỉnh alpha
        Label(control_frame, text="Alpha (quy mô không gian):", bg='#f0f0f0').grid(row=0, column=1, padx=5, pady=10)
        self.alpha_scale = Scale(control_frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, 
                                length=200, command=self.update_alpha)
        self.alpha_scale.set(self.alpha)
        self.alpha_scale.grid(row=0, column=2, padx=5, pady=10)
        
        # Thanh điều chỉnh số lần lặp
        Label(control_frame, text="Số lần lặp (k):", bg='#f0f0f0').grid(row=0, column=3, padx=5, pady=10)
        self.k_scale = Scale(control_frame, from_=1, to=10, resolution=1, orient=tk.HORIZONTAL, 
                            length=200, command=self.update_k)
        self.k_scale.set(self.k_iterations)
        self.k_scale.grid(row=0, column=4, padx=5, pady=10)
        
        # Nút áp dụng bộ lọc
        Button(control_frame, text="Áp dụng bộ lọc", command=self.apply_filter,
               bg="#2196F3", fg="white", font=("Arial", 12), padx=10).grid(row=0, column=5, padx=10, pady=10)
        
        # Nút lưu hình ảnh
        Button(control_frame, text="Lưu kết quả", command=self.save_image,
               bg="#FF9800", fg="white", font=("Arial", 12), padx=10).grid(row=0, column=6, padx=10, pady=10)
        
        # Frame chứa hình ảnh
        self.image_frame = Frame(self.root, bg='#e0e0e0')
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Label hiển thị hình ảnh gốc
        self.original_label = Label(self.image_frame, bg='#e0e0e0', text="Hình ảnh gốc")
        self.original_label.grid(row=0, column=0, padx=10, pady=5)
        
        # Label hiển thị hình ảnh kết quả
        self.result_label = Label(self.image_frame, bg='#e0e0e0', text="Kết quả")
        self.result_label.grid(row=0, column=1, padx=10, pady=5)
        
        # Canvas hiển thị hình ảnh gốc
        self.original_canvas = Label(self.image_frame, bg='black', width=500, height=500)
        self.original_canvas.grid(row=1, column=0, padx=10, pady=10)
        
        # Canvas hiển thị hình ảnh kết quả
        self.result_canvas = Label(self.image_frame, bg='black', width=500, height=500)
        self.result_canvas.grid(row=1, column=1, padx=10, pady=10)
        
        # Frame chứa thông tin trạng thái
        status_frame = Frame(self.root, bg='#f0f0f0', height=30)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Label hiển thị trạng thái
        self.status_label = Label(status_frame, text="Sẵn sàng", bg='#f0f0f0')
        self.status_label.pack(side=tk.LEFT, padx=10)
    
    def update_alpha(self, value):
        self.alpha = float(value)
    
    def update_k(self, value):
        self.k_iterations = int(value)
    
    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("All Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("BMP files", "*.bmp"),
                ("TIFF files", "*.tif *.tiff"),
                ("All Files", "*.*")
            ],
        )
        
        if file_path:
            self.current_image_path = file_path
            self.status_label.config(text=f"Đã tải hình ảnh: {os.path.basename(file_path)}")
            
            # Đọc hình ảnh bằng OpenCV
            self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            # Hiển thị hình ảnh gốc
            self.display_image(self.original_image, self.original_canvas, "Hình ảnh gốc")
            
            # Xóa hình ảnh kết quả cũ nếu có
            self.result_canvas.config(image='')
            self.result_canvas.image = None
            self.filtered_image = None
    
    def apply_filter(self):
        if self.original_image is None:
            self.status_label.config(text="Chưa có hình ảnh nào được tải lên!")
            return
        
        self.status_label.config(text="Đang xử lý...")
        self.root.update()
        
        try:
            # Áp dụng bộ lọc bảo toàn cạnh
            self.filtered_image = self.edge_preserve_filter(
                self.original_image,
                k=self.k_iterations,
                alpha=self.alpha,
                kernel_size=self.kernel_size
            )
            
            # Hiển thị hình ảnh kết quả
            self.display_image(self.filtered_image, self.result_canvas, "Kết quả")
            
            self.status_label.config(text="Đã áp dụng bộ lọc thành công!")
        except Exception as e:
            self.status_label.config(text=f"Lỗi: {str(e)}")
    
    def edge_preserve_filter(self, image, k=5, alpha=0.5, kernel_size=11):
        # Bước 1: Lọc hình ảnh với bộ lọc phân cụm
        starting_image, beta = get_starting_point_v2(image, alpha, kernel_size=kernel_size)
        filtered_image = image.copy()
        
        for i in range(k):
            filtered_image = cluster_filter_v3(
                image, starting_image, alpha, beta, k=1, kernel_size=kernel_size
            )
            starting_image = filtered_image.copy()
        
        # Bước 2: Tính hiệu số
        Image_i = filtered_image
        Image_d = image - Image_i
        
        # Bước 3: Tính trung bình và phương sai cục bộ
        M, V = compute_local_mean_var(Image_d)
        
        # Bước 4: Tạo mặt nạ thông qua ngưỡng
        threshold = V * 2.5
        Image_m = np.where(np.abs(Image_d - M) < threshold, Image_i, image)
        
        # Bước 5: Trừ mặt nạ từ ảnh gốc
        s = 0.5
        Image_o = image - s * Image_m
        
        return Image_o
    
    def display_image(self, image, canvas, title=""):
        if image is None:
            return
        
        # Chuyển đổi hình ảnh OpenCV sang định dạng PIL
        image_pil = Image.fromarray(image)
        
        # Điều chỉnh kích thước để vừa với canvas
        width, height = image_pil.size
        max_size = 500
        scale = min(max_size / width, max_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image_pil = image_pil.resize((new_width, new_height))
        
        # Chuyển đổi sang ImageTk
        image_tk = ImageTk.PhotoImage(image=image_pil)
        
        # Hiển thị lên canvas
        canvas.config(image=image_tk, width=new_width, height=new_height)
        canvas.image = image_tk  # Giữ tham chiếu để tránh bị thu hồi bởi garbage collector
    
    def save_image(self):
        if self.filtered_image is None:
            self.status_label.config(text="Chưa có hình ảnh kết quả nào để lưu!")
            return
        
        # Hiển thị hộp thoại lưu file
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Lưu hình ảnh
                cv2.imwrite(file_path, self.filtered_image)
                self.status_label.config(text=f"Đã lưu hình ảnh tại: {file_path}")
            except Exception as e:
                self.status_label.config(text=f"Lỗi khi lưu hình ảnh: {str(e)}")

# Khởi chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = EdgePreservingFilterDemo(root)
    root.mainloop()