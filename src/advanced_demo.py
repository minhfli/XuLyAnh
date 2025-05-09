import tkinter as tk
from tkinter import ttk, filedialog, Scale, Label, Button, Frame
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# Thêm đường dẫn hiện tại vào sys.path để có thể import module filter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.filter import get_starting_point_v2, cluster_filter_v3, compute_local_mean_var

class AdvancedFilterDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Bộ lọc bảo toàn cạnh - Demo nâng cao")
        self.root.geometry("1280x800")
        
        # Biến lưu trữ hình ảnh và tham số
        self.original_image = None
        self.current_image_path = None
        self.alpha = 0.5
        self.k_iterations = 5
        self.kernel_size = 11
        
        # Biến lưu trữ kết quả các phương pháp lọc
        self.gaussian_result = None
        self.cluster_result = None
        self.edge_preserve_result = None
        
        # Tạo giao diện
        self.create_ui()
    
    def create_ui(self):
        # Tạo frame chứa điều khiển
        control_frame = Frame(self.root, bg='#f0f0f0')
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Tạo frame chứa nút điều khiển
        button_frame = Frame(control_frame, bg='#f0f0f0')
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Nút tải hình ảnh
        Button(button_frame, text="Tải hình ảnh", command=self.open_image, 
               bg="#4CAF50", fg="white", font=("Arial", 12), padx=10).pack(side=tk.LEFT, padx=10)
        
        # Nút áp dụng bộ lọc
        Button(button_frame, text="Áp dụng bộ lọc", command=self.apply_filters,
               bg="#2196F3", fg="white", font=("Arial", 12), padx=10).pack(side=tk.LEFT, padx=10)
        
        # Nút lưu kết quả
        Button(button_frame, text="Lưu kết quả", command=self.save_comparison,
               bg="#FF9800", fg="white", font=("Arial", 12), padx=10).pack(side=tk.LEFT, padx=10)
        
        # Tạo frame chứa các thanh điều chỉnh
        slider_frame = Frame(control_frame, bg='#f0f0f0')
        slider_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Thanh điều chỉnh alpha
        Label(slider_frame, text="Alpha (quy mô không gian):", bg='#f0f0f0').grid(row=0, column=0, padx=5, pady=5)
        self.alpha_scale = Scale(slider_frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, 
                                length=200, command=self.update_alpha)
        self.alpha_scale.set(self.alpha)
        self.alpha_scale.grid(row=0, column=1, padx=5, pady=5)
        
        # Thanh điều chỉnh số lần lặp
        Label(slider_frame, text="Số lần lặp (k):", bg='#f0f0f0').grid(row=0, column=2, padx=5, pady=5)
        self.k_scale = Scale(slider_frame, from_=1, to=10, resolution=1, orient=tk.HORIZONTAL, 
                             length=200, command=self.update_k)
        self.k_scale.set(self.k_iterations)
        self.k_scale.grid(row=0, column=3, padx=5, pady=5)
        
        # Tạo frame chứa hình ảnh
        self.image_frame = Frame(self.root, bg='#e0e0e0')
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tạo notebook để chứa các tab
        self.notebook = ttk.Notebook(self.image_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tạo tab hiển thị hình ảnh gốc và kết quả
        self.comparison_tab = Frame(self.notebook, bg='#e0e0e0')
        self.notebook.add(self.comparison_tab, text="So sánh kết quả")
        
        # Tạo đồ thị Matplotlib để hiển thị hình ảnh
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.tight_layout(pad=3.0)
        
        # Tạo canvas Matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.comparison_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tạo tab hiển thị các bước xử lý
        self.steps_tab = Frame(self.notebook, bg='#e0e0e0')
        self.notebook.add(self.steps_tab, text="Các bước xử lý")
        
        # Tạo đồ thị Matplotlib cho tab "Các bước xử lý"
        self.steps_fig, self.steps_axes = plt.subplots(2, 3, figsize=(12, 8))
        self.steps_fig.tight_layout(pad=3.0)
        
        # Tạo canvas Matplotlib cho tab "Các bước xử lý"
        self.steps_canvas = FigureCanvasTkAgg(self.steps_fig, master=self.steps_tab)
        self.steps_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tạo label trạng thái
        self.status_label = Label(self.root, text="Sẵn sàng", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
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
            self.display_comparison()
    
    def apply_filters(self):
        if self.original_image is None:
            self.status_label.config(text="Chưa có hình ảnh nào được tải lên!")
            return
        
        self.status_label.config(text="Đang xử lý...")
        self.root.update()
        
        # Sử dụng thread để không làm đơ giao diện
        threading.Thread(target=self._apply_filters_thread).start()
    
    def _apply_filters_thread(self):
        try:
            # Áp dụng bộ lọc Gaussian
            self.gaussian_result = cv2.GaussianBlur(self.original_image, (self.kernel_size, self.kernel_size), 0)
            
            # Áp dụng bộ lọc phân cụm
            starting_image, beta = get_starting_point_v2(
                self.original_image, alpha=self.alpha, kernel_size=self.kernel_size
            )
            self.cluster_result = cluster_filter_v3(
                self.original_image, starting_image, k=self.k_iterations, 
                beta=beta, alpha=self.alpha, kernel_size=self.kernel_size
            )
            
            # Áp dụng bộ lọc bảo toàn cạnh hoàn chỉnh
            self.edge_preserve_result = self.edge_preserve_filter(
                self.original_image, k=self.k_iterations, alpha=self.alpha, kernel_size=self.kernel_size
            )
            
            # Cập nhật giao diện
            self.root.after(0, self.display_comparison)
            self.root.after(0, self.display_steps)
            self.root.after(0, lambda: self.status_label.config(text="Đã áp dụng bộ lọc thành công!"))
        
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Lỗi: {str(e)}"))
    
    def edge_preserve_filter(self, image, k=5, alpha=0.5, kernel_size=11):
        # Bước 1: Lọc hình ảnh với bộ lọc phân cụm
        starting_image, beta = get_starting_point_v2(image, alpha, kernel_size=kernel_size)
        filtered_image = starting_image.copy()
        
        for i in range(k):
            filtered_image = cluster_filter_v3(
                image, filtered_image, alpha, beta, k=1, kernel_size=kernel_size
            )
        
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
        
        # Lưu các hình ảnh trung gian để hiển thị các bước
        self.step_images = {
            "original": image,
            "filtered": filtered_image,
            "diff": Image_d,
            "mask": Image_m,
            "result": Image_o
        }
        
        return Image_o
    
    def display_comparison(self):
        # Xóa các trục
        for ax in self.axes.flat:
            ax.clear()
        
        # Hiển thị hình ảnh gốc
        if self.original_image is not None:
            self.axes[0, 0].imshow(self.original_image, cmap='gray')
            self.axes[0, 0].set_title('Hình ảnh gốc')
            self.axes[0, 0].axis('off')
        
        # Hiển thị kết quả bộ lọc Gaussian
        if self.gaussian_result is not None:
            self.axes[0, 1].imshow(self.gaussian_result, cmap='gray')
            self.axes[0, 1].set_title('Bộ lọc Gaussian')
            self.axes[0, 1].axis('off')
        
        # Hiển thị kết quả bộ lọc phân cụm
        if self.cluster_result is not None:
            self.axes[1, 0].imshow(self.cluster_result, cmap='gray')
            self.axes[1, 0].set_title('Bộ lọc phân cụm')
            self.axes[1, 0].axis('off')
        
        # Hiển thị kết quả bộ lọc bảo toàn cạnh
        if self.edge_preserve_result is not None:
            self.axes[1, 1].imshow(self.edge_preserve_result, cmap='gray')
            self.axes[1, 1].set_title('Bộ lọc bảo toàn cạnh')
            self.axes[1, 1].axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def display_steps(self):
        if not hasattr(self, 'step_images'):
            return
        
        # Xóa các trục
        for ax in self.steps_axes.flat:
            ax.clear()
        
        # Hiển thị hình ảnh gốc
        self.steps_axes[0, 0].imshow(self.step_images["original"], cmap='gray')
        self.steps_axes[0, 0].set_title('Hình ảnh gốc')
        self.steps_axes[0, 0].axis('off')
        
        # Hiển thị hình ảnh sau khi lọc phân cụm
        self.steps_axes[0, 1].imshow(self.step_images["filtered"], cmap='gray')
        self.steps_axes[0, 1].set_title('Bước 1: Lọc phân cụm')
        self.steps_axes[0, 1].axis('off')
        
        # Hiển thị hình ảnh hiệu số
        self.steps_axes[0, 2].imshow(self.step_images["diff"], cmap='gray')
        self.steps_axes[0, 2].set_title('Bước 2: Hiệu số')
        self.steps_axes[0, 2].axis('off')
        
        # Hiển thị hình ảnh mặt nạ
        self.steps_axes[1, 0].imshow(self.step_images["mask"], cmap='gray')
        self.steps_axes[1, 0].set_title('Bước 4: Mặt nạ')
        self.steps_axes[1, 0].axis('off')
        
        # Hiển thị hình ảnh kết quả
        self.steps_axes[1, 1].imshow(self.step_images["result"], cmap='gray')
        self.steps_axes[1, 1].set_title('Bước 5: Kết quả')
        self.steps_axes[1, 1].axis('off')
        
        # Hiển thị biểu đồ histogram của hình ảnh kết quả
        self.steps_axes[1, 2].hist(self.step_images["result"].flatten(), bins=50, color='blue', alpha=0.7)
        self.steps_axes[1, 2].set_title('Histogram kết quả')
        
        self.steps_fig.tight_layout()
        self.steps_canvas.draw()
    
    def save_comparison(self):
        if self.original_image is None or self.edge_preserve_result is None:
            self.status_label.config(text="Chưa có kết quả để lưu!")
            return
        
        # Hiển thị hộp thoại lưu file
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Tạo hình ảnh so sánh
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 2, 1)
                plt.imshow(self.original_image, cmap='gray')
                plt.title('Hình ảnh gốc')
                plt.axis('off')
                
                plt.subplot(2, 2, 2)
                plt.imshow(self.gaussian_result, cmap='gray')
                plt.title('Bộ lọc Gaussian')
                plt.axis('off')
                
                plt.subplot(2, 2, 3)
                plt.imshow(self.cluster_result, cmap='gray')
                plt.title('Bộ lọc phân cụm')
                plt.axis('off')
                
                plt.subplot(2, 2, 4)
                plt.imshow(self.edge_preserve_result, cmap='gray')
                plt.title('Bộ lọc bảo toàn cạnh')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(file_path, dpi=300)
                plt.close()
                
                self.status_label.config(text=f"Đã lưu hình ảnh so sánh tại: {file_path}")
            except Exception as e:
                self.status_label.config(text=f"Lỗi khi lưu hình ảnh: {str(e)}")

# Khởi chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedFilterDemo(root)
    root.mainloop()