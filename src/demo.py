import tkinter as tk
from tkinter import filedialog, Scale, Label, Button, Frame, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from plot_helper import plot_img, axs

# Thêm đường dẫn hiện tại vào sys.path để có thể import module filter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.filter import (
    enhance_with_cluster_filter, 
    enhance_with_gausian_filter, 
    get_starting_point_v2, 
    cluster_filter_v3, 
)

class EdgePreservingFilterDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Bộ lọc bảo toàn cạnh - Demo")
        self.root.geometry("1200x800")
        
        # Hiển thị cửa sổ ở giữa màn hình
        self.center_window(self.root)
        
        # Biến lưu trữ hình ảnh và tham số
        self.original_image = None
        self.filtered_image = None
        self.current_image_path = None
        self.alpha = 0.5
        self.k_iterations = 5
        self.kernel_size = 11
        
        # Biến kiểm soát luồng
        self.processing = False
        
        # Tạo giao diện
        self.create_ui()
    
    def center_window(self, window):
        """Đặt cửa sổ ở giữa màn hình"""
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    def create_ui(self):
        # Frame chính
        main_frame = Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Tạo cấu trúc cho giao diện với layout dạng lưới
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=0)  # Phần điều khiển
        main_frame.grid_rowconfigure(1, weight=1)  # Phần hình ảnh
        main_frame.grid_rowconfigure(2, weight=0)  # Phần trạng thái
        
        # Frame chứa điều khiển
        control_frame = Frame(main_frame, bg='#f0f0f0', padx=10, pady=10)
        control_frame.grid(row=0, column=0, columnspan=2, sticky='nsew')
        
        # Frame cho các nút
        button_frame = Frame(control_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, pady=5)
        
        # Nút tải hình ảnh
        Button(button_frame, text="Tải hình ảnh", command=self.open_image, 
               bg="#4CAF50", fg="white", font=("Arial", 12), padx=10).pack(side=tk.LEFT, padx=5)
        
        # Thanh điều chỉnh alpha
        Label(button_frame, text="Alpha:", bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
        self.alpha_scale = Scale(button_frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, 
                                length=150, command=self.update_alpha)
        self.alpha_scale.set(self.alpha)
        self.alpha_scale.pack(side=tk.LEFT, padx=5)
        
        # Thanh điều chỉnh số lần lặp
        Label(button_frame, text="Số lần lặp (k):", bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
        self.k_scale = Scale(button_frame, from_=1, to=10, resolution=1, orient=tk.HORIZONTAL, 
                            length=150, command=self.update_k)
        self.k_scale.set(self.k_iterations)
        self.k_scale.pack(side=tk.LEFT, padx=5)
        
        # Nút áp dụng các loại bộ lọc
        process_frame = Frame(control_frame, bg='#f0f0f0')
        process_frame.pack(fill=tk.X, pady=5)
        
        Button(process_frame, text="Áp dụng bộ lọc Gaussian", command=self.apply_gaussian_filter,
               bg="#2196F3", fg="white", font=("Arial", 12), padx=10).pack(side=tk.LEFT, padx=5)
        
        Button(process_frame, text="Áp dụng bộ lọc phân cụm", command=self.apply_cluster_filter,
               bg="#FF9800", fg="white", font=("Arial", 12), padx=10).pack(side=tk.LEFT, padx=5)
        
        Button(process_frame, text="Áp dụng bộ lọc bảo toàn cạnh", command=self.apply_edge_preserving_filter,
               bg="#9C27B0", fg="white", font=("Arial", 12), padx=10).pack(side=tk.LEFT, padx=5)
        
        # Nút lưu kết quả
        Button(process_frame, text="Lưu kết quả", command=self.save_image,
               bg="#607D8B", fg="white", font=("Arial", 12), padx=10).pack(side=tk.RIGHT, padx=5)
        
        # Thanh tiến trình
        progress_frame = Frame(control_frame, bg='#f0f0f0')
        progress_frame.pack(fill=tk.X, pady=5)
        
        Label(progress_frame, text="Tiến trình:", bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
        self.progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=600, mode='determinate')
        self.progress.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Tạo các frame để hiển thị hình ảnh
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.subplots_adjust(wspace=0.05)
        self.canvas_frame = Frame(main_frame, bg='#e0e0e0')
        self.canvas_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=10, pady=10)
        
        # Tạo canvas matplotlib để hiển thị hình ảnh
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Chuẩn bị các trục
        self.ax[0].set_title("Hình ảnh gốc")
        self.ax[0].axis('off')
        self.ax[1].set_title("Kết quả")
        self.ax[1].axis('off')
        
        # Frame chứa thông tin trạng thái
        status_frame = Frame(main_frame, bg='#f0f0f0', height=30)
        status_frame.grid(row=2, column=0, columnspan=2, sticky='ew')
        
        # Label hiển thị trạng thái
        self.status_label = Label(status_frame, text="Sẵn sàng", bg='#f0f0f0', anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=10, fill=tk.X)
    
    def update_alpha(self, value):
        self.alpha = float(value)
    
    def update_k(self, value):
        self.k_iterations = int(value)
    
    def open_image(self):
        file_types = [
            ("All Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.gif"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("BMP files", "*.bmp"),
            ("TIFF files", "*.tif *.tiff"),
            ("All Files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            parent=self.root,
            filetypes=file_types,
            initialdir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")
        )
        
        if file_path:
            self.current_image_path = file_path
            self.status_label.config(text=f"Đã tải hình ảnh: {os.path.basename(file_path)}")
            
            # Đọc hình ảnh bằng OpenCV
            self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            # Hiển thị hình ảnh gốc
            self.display_images(self.original_image, self.original_image)
            
            # Reset biến
            self.filtered_image = None
            
            # Reset tiến trình
            self.progress['value'] = 0
    
    def apply_gaussian_filter(self):
        if self.original_image is None:
            self.status_label.config(text="Chưa có hình ảnh nào được tải lên!")
            return
        
        if self.processing:
            self.status_label.config(text="Đang xử lý, vui lòng đợi...")
            return
        
        # Bắt đầu xử lý trong một luồng riêng biệt
        self.processing = True
        threading.Thread(target=self._apply_gaussian_filter_thread).start()
    
    def _apply_gaussian_filter_thread(self):
        try:
            self.status_label.config(text="Đang áp dụng bộ lọc Gaussian...")
            self.progress['value'] = 0
            self.root.update()
            
            # Áp dụng bộ lọc Gaussian
            self.filtered_image = enhance_with_gausian_filter(
                self.original_image,
                k=self.k_iterations,
                alpha=self.alpha,
                kernel_size=self.kernel_size
            )
            
            # Hiển thị kết quả
            self.root.after(0, lambda: self.display_images(self.original_image, self.filtered_image))
            self.root.after(0, lambda: self.status_label.config(text="Đã áp dụng bộ lọc Gaussian thành công!"))
            self.root.after(0, lambda: self.progress.config(value=100))
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Lỗi: {str(e)}"))
        finally:
            self.processing = False
    
    def apply_cluster_filter(self):
        if self.original_image is None:
            self.status_label.config(text="Chưa có hình ảnh nào được tải lên!")
            return
        
        if self.processing:
            self.status_label.config(text="Đang xử lý, vui lòng đợi...")
            return
        
        # Bắt đầu xử lý trong một luồng riêng biệt
        self.processing = True
        threading.Thread(target=self._apply_cluster_filter_thread).start()
    
    def _apply_cluster_filter_thread(self):
        try:
            self.status_label.config(text="Đang áp dụng bộ lọc phân cụm...")
            self.progress['value'] = 0
            self.root.update()
            
            # Áp dụng bộ lọc phân cụm
            starting_image, beta = get_starting_point_v2(
                self.original_image, alpha=self.alpha, kernel_size=self.kernel_size
            )
            self.progress['value'] = 30
            self.root.update()
            
            self.filtered_image = cluster_filter_v3(
                self.original_image, starting_image, k=self.k_iterations, 
                beta=beta, alpha=self.alpha, kernel_size=self.kernel_size
            )
            
            # Hiển thị kết quả
            self.root.after(0, lambda: self.display_images(self.original_image, self.filtered_image))
            self.root.after(0, lambda: self.status_label.config(text="Đã áp dụng bộ lọc phân cụm thành công!"))
            self.root.after(0, lambda: self.progress.config(value=100))
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Lỗi: {str(e)}"))
        finally:
            self.processing = False
    
    def apply_edge_preserving_filter(self):
        if self.original_image is None:
            self.status_label.config(text="Chưa có hình ảnh nào được tải lên!")
            return
        
        if self.processing:
            self.status_label.config(text="Đang xử lý, vui lòng đợi...")
            return
        
        # Bắt đầu xử lý trong một luồng riêng biệt
        self.processing = True
        threading.Thread(target=self._apply_edge_preserving_filter_thread).start()
    
    def _apply_edge_preserving_filter_thread(self):
        try:
            self.status_label.config(text="Đang áp dụng bộ lọc bảo toàn cạnh...")
            self.progress['value'] = 0
            self.root.update()
            
            # Áp dụng bộ lọc bảo toàn cạnh
            self.filtered_image = enhance_with_cluster_filter(
                self.original_image,
                k=self.k_iterations,
                alpha=self.alpha,
                kernel_size=self.kernel_size
            )
            
            # Hiển thị kết quả
            self.root.after(0, lambda: self.display_images(self.original_image, self.filtered_image))
            self.root.after(0, lambda: self.status_label.config(text="Đã áp dụng bộ lọc bảo toàn cạnh thành công!"))
            self.root.after(0, lambda: self.progress.config(value=100))
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Lỗi: {str(e)}"))
        finally:
            self.processing = False
    
    def display_images(self, img1, img2):
        # Xóa nội dung trục hiện tại
        self.ax[0].clear()
        self.ax[1].clear()
        
        # Hiển thị hình ảnh gốc
        self.ax[0].imshow(img1, cmap='gray')
        self.ax[0].set_title("Hình ảnh gốc")
        self.ax[0].axis('off')
        plot_img(self.ax[0], img1, "Hình ảnh gốc", cmap=plt.get_cmap("gray"))
        
        # Hiển thị hình ảnh kết quả
        self.ax[1].imshow(img2, cmap='gray')
        self.ax[1].set_title("Kết quả")
        self.ax[1].axis('off')
        plot_img(self.ax[1], img2, "Kết quả", cmap=plt.get_cmap("gray"), vmax=None, vmin=None)

        # Cập nhật canvas
        self.canvas.draw()
    
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