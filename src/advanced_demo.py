import tkinter as tk
from tkinter import filedialog, Scale, Label, Button, Frame, ttk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
from plot_helper import plot_img

# Thêm đường dẫn hiện tại vào sys.path để có thể import module filter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.filter import (
        enhance_with_cluster_filter, 
        enhance_with_gausian_filter, 
        get_starting_point_v2, 
        cluster_filter_v3, 
        compute_local_mean_var
    )
except ImportError:
    messagebox.showerror("Lỗi Import", "Không thể import các hàm từ module filter.py. Vui lòng kiểm tra cấu trúc thư mục.")
    sys.exit(1)

class AdvancedFilterDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Bộ lọc bảo toàn cạnh - Demo nâng cao")
        self.root.geometry("1280x800")
        self.root.attributes('-fullscreen', True)
        
        # Hiển thị cửa sổ ở giữa màn hình
        self.center_window(self.root)
        
        # Biến lưu trữ hình ảnh và tham số
        self.original_image = None
        self.current_image_path = None
        self.alpha = 0.5
        self.k_iterations = 5
        self.kernel_size = 11
        self.local_kernel_size = 40
        self.s_factor = 0.5  # Hệ số s trong công thức Image_o = image - s * Image_m
        
        # Kết quả các loại bộ lọc
        self.gaussian_result = None
        self.cluster_result = None
        self.edge_preserve_result = None
        
        # Kết quả các bước trung gian
        self.intermediate_results = {}
        
        # Biến kiểm soát luồng
        self.processing = False
        
        # Tạo giao diện
        self.create_ui()
        
        # Kết nối sự kiện đóng cửa sổ
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def center_window(self, window):
        """Đặt cửa sổ ở giữa màn hình"""
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    def create_ui(self):
        # Tạo thanh menu
        menubar = tk.Menu(self.root)
        
        # Menu File
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Mở hình ảnh", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Lưu kết quả", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Thoát", command=self.on_closing)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Menu Filter
        filter_menu = tk.Menu(menubar, tearoff=0)
        filter_menu.add_command(label="Bộ lọc Gaussian", command=self.apply_gaussian_filter)
        filter_menu.add_command(label="Bộ lọc phân cụm", command=self.apply_cluster_filter)
        filter_menu.add_command(label="Bộ lọc bảo toàn cạnh", command=self.apply_edge_preserving_filter)
        filter_menu.add_separator()
        filter_menu.add_command(label="So sánh các bộ lọc", command=self.compare_filters)
        menubar.add_cascade(label="Bộ lọc", menu=filter_menu)
        
        # Menu View
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Hiển thị các bước trung gian", command=self.show_intermediate_steps)
        view_menu.add_command(label="Hiển thị biểu đồ histogram", command=self.show_histogram)
        menubar.add_cascade(label="Xem", menu=view_menu)
        
        # Menu Help
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Hướng dẫn sử dụng", command=self.show_help)
        help_menu.add_command(label="Thông tin", command=self.show_about)
        menubar.add_cascade(label="Trợ giúp", menu=help_menu)
        
        self.root.config(menu=menubar)
        
        # Tạo Notebook (tab)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Tổng quan
        self.overview_tab = Frame(self.notebook)
        self.notebook.add(self.overview_tab, text="Tổng quan")
        
        # Tab 2: Các bước trung gian
        self.steps_tab = Frame(self.notebook)
        self.notebook.add(self.steps_tab, text="Các bước trung gian")
        
        # Tab 3: So sánh các bộ lọc
        self.comparison_tab = Frame(self.notebook)
        self.notebook.add(self.comparison_tab, text="So sánh các bộ lọc")
        
        # Tab 4: Phân tích histogram
        self.histogram_tab = Frame(self.notebook)
        self.notebook.add(self.histogram_tab, text="Phân tích histogram")
        
        # Thiết lập nội dung tab "Tổng quan"
        self.setup_overview_tab()
        
        # Thiết lập nội dung tab "Các bước trung gian"
        self.setup_steps_tab()
        
        # Thiết lập nội dung tab "So sánh các bộ lọc"
        self.setup_comparison_tab()
        
        # Thiết lập nội dung tab "Phân tích histogram"
        self.setup_histogram_tab()
        
        # Frame chứa thông tin trạng thái
        status_frame = Frame(self.root, bg='#f0f0f0', height=30)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Thanh tiến trình
        self.progress = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, length=600, mode='determinate')
        self.progress.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Label hiển thị trạng thái
        self.status_label = Label(status_frame, text="Sẵn sàng", bg='#f0f0f0')
        self.status_label.pack(side=tk.LEFT, padx=10)
    
    def setup_overview_tab(self):
        # Frame cho điều khiển
        control_frame = Frame(self.overview_tab, bg='#f0f0f0')
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Frame cho các nút
        button_frame = Frame(control_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, pady=5)
        
        # Nút tải hình ảnh
        Button(button_frame, text="Tải hình ảnh", command=self.open_image, 
               bg="#4CAF50", fg="white", font=("Arial", 12), padx=10).pack(side=tk.LEFT, padx=5)
        
        # Nút áp dụng bộ lọc bảo toàn cạnh
        Button(button_frame, text="Áp dụng bộ lọc bảo toàn cạnh", command=self.apply_edge_preserving_filter,
               bg="#FF9800", fg="white", font=("Arial", 12), padx=10).pack(side=tk.LEFT, padx=5)
        
        # Nút lưu kết quả
        Button(button_frame, text="Lưu kết quả", command=self.save_image,
               bg="#2196F3", fg="white", font=("Arial", 12), padx=10).pack(side=tk.RIGHT, padx=5)
        
        # Frame cho các tham số
        param_frame = Frame(control_frame, bg='#f0f0f0')
        param_frame.pack(fill=tk.X, pady=5)
        
        # Tạo LabelFrame để nhóm các tham số
        param_group = ttk.LabelFrame(param_frame, text="Tham số bộ lọc")
        param_group.pack(fill=tk.X, pady=5, padx=5)
        
        # Grid layout cho các tham số
        param_group.columnconfigure(0, weight=0)  # Label
        param_group.columnconfigure(1, weight=1)  # Slider
        param_group.columnconfigure(2, weight=0)  # Value
        
        # Tham số alpha
        ttk.Label(param_group, text="Alpha:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.alpha_scale = ttk.Scale(param_group, from_=0.1, to=1.0, value=self.alpha, 
                                      command=self.update_alpha)
        self.alpha_scale.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.alpha_value = ttk.Label(param_group, text=f"{self.alpha:.1f}")
        self.alpha_value.grid(row=0, column=2, padx=5, pady=5)
        
        # Tham số k
        ttk.Label(param_group, text="Số lần lặp (k):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.k_scale = ttk.Scale(param_group, from_=1, to=10, value=self.k_iterations, 
                                  command=self.update_k)
        self.k_scale.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        self.k_value = ttk.Label(param_group, text=f"{self.k_iterations}")
        self.k_value.grid(row=1, column=2, padx=5, pady=5)
        
        # Tham số kernel size
        ttk.Label(param_group, text="Kích thước kernel:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.kernel_scale = ttk.Scale(param_group, from_=3, to=21, value=self.kernel_size, 
                                       command=self.update_kernel_size)
        self.kernel_scale.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        self.kernel_value = ttk.Label(param_group, text=f"{self.kernel_size}")
        self.kernel_value.grid(row=2, column=2, padx=5, pady=5)
        
        # Tham số local kernel size
        ttk.Label(param_group, text="Kích thước kernel cục bộ:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.local_kernel_scale = ttk.Scale(param_group, from_=10, to=100, value=self.local_kernel_size, 
                                             command=self.update_local_kernel_size)
        self.local_kernel_scale.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)
        self.local_kernel_value = ttk.Label(param_group, text=f"{self.local_kernel_size}")
        self.local_kernel_value.grid(row=3, column=2, padx=5, pady=5)
        
        # Tham số s
        ttk.Label(param_group, text="Hệ số s:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.s_scale = ttk.Scale(param_group, from_=0.1, to=1.0, value=self.s_factor, 
                                  command=self.update_s_factor)
        self.s_scale.grid(row=4, column=1, padx=5, pady=5, sticky=tk.EW)
        self.s_value = ttk.Label(param_group, text=f"{self.s_factor:.1f}")
        self.s_value.grid(row=4, column=2, padx=5, pady=5)
        
        # Frame cho hình ảnh
        image_frame = Frame(self.overview_tab)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tạo đồ thị matplotlib cho tab "Tổng quan"
        self.overview_fig, self.overview_ax = plt.subplots(1, 2, figsize=(12, 6))
        self.overview_fig.subplots_adjust(wspace=0.05)
        
        # Chuẩn bị các trục
        self.overview_ax[0].set_title("Hình ảnh gốc")
        self.overview_ax[0].axis('off')
        self.overview_ax[1].set_title("Kết quả")
        self.overview_ax[1].axis('off')
        
        # Tạo canvas matplotlib
        self.overview_canvas = FigureCanvasTkAgg(self.overview_fig, master=image_frame)
        self.overview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_steps_tab(self):
        # Frame cho đồ thị
        steps_frame = Frame(self.steps_tab)
        steps_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tạo đồ thị matplotlib cho tab "Các bước trung gian"
        self.steps_fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 3)
        
        # Tạo các trục con
        self.steps_axes = {
            'original': self.steps_fig.add_subplot(gs[0, 0]),
            'filtered': self.steps_fig.add_subplot(gs[0, 1]),
            'diff': self.steps_fig.add_subplot(gs[0, 2]),
            'local_mean': self.steps_fig.add_subplot(gs[1, 0]),
            'mask': self.steps_fig.add_subplot(gs[1, 1]),
            'result': self.steps_fig.add_subplot(gs[1, 2])
        }
        
        # Đặt tiêu đề cho các trục
        self.steps_axes['original'].set_title("Hình ảnh gốc")
        self.steps_axes['original'].axis('off')
        self.steps_axes['filtered'].set_title("Bước 1: Lọc phân cụm")
        self.steps_axes['filtered'].axis('off')
        self.steps_axes['diff'].set_title("Bước 2: Hiệu số")
        self.steps_axes['diff'].axis('off')
        self.steps_axes['local_mean'].set_title("Bước 3: Trung bình cục bộ")
        self.steps_axes['local_mean'].axis('off')
        self.steps_axes['mask'].set_title("Bước 4: Mặt nạ")
        self.steps_axes['mask'].axis('off')
        self.steps_axes['result'].set_title("Bước 5+6: Kết quả")
        self.steps_axes['result'].axis('off')
        
        # Điều chỉnh khoảng cách giữa các trục
        self.steps_fig.tight_layout()
        
        # Tạo canvas matplotlib
        self.steps_canvas = FigureCanvasTkAgg(self.steps_fig, master=steps_frame)
        self.steps_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_comparison_tab(self):
        # Frame cho đồ thị
        comparison_frame = Frame(self.comparison_tab)
        comparison_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tạo đồ thị matplotlib cho tab "So sánh các bộ lọc"
        self.comparison_fig, self.comparison_ax = plt.subplots(2, 2, figsize=(12, 8))
        self.comparison_fig.tight_layout(pad=3.0)
        
        # Đặt tiêu đề cho các trục
        self.comparison_ax[0, 0].set_title("Hình ảnh gốc")
        self.comparison_ax[0, 0].axis('off')
        self.comparison_ax[0, 1].set_title("Bộ lọc Gaussian")
        self.comparison_ax[0, 1].axis('off')
        self.comparison_ax[1, 0].set_title("Bộ lọc phân cụm")
        self.comparison_ax[1, 0].axis('off')
        self.comparison_ax[1, 1].set_title("Bộ lọc bảo toàn cạnh")
        self.comparison_ax[1, 1].axis('off')
        
        # Tạo canvas matplotlib
        self.comparison_canvas = FigureCanvasTkAgg(self.comparison_fig, master=comparison_frame)
        self.comparison_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Nút so sánh
        Button(comparison_frame, text="So sánh các bộ lọc", command=self.compare_filters,
               bg="#9C27B0", fg="white", font=("Arial", 12), padx=10).pack(pady=10)
    
    def setup_histogram_tab(self):
        # Frame cho đồ thị
        histogram_frame = Frame(self.histogram_tab)
        histogram_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tạo đồ thị matplotlib cho tab "Phân tích histogram"
        self.histogram_fig, self.histogram_ax = plt.subplots(2, 2, figsize=(12, 8))
        self.histogram_fig.tight_layout(pad=3.0)
        
        # Đặt tiêu đề cho các trục
        self.histogram_ax[0, 0].set_title("Hình ảnh gốc")
        self.histogram_ax[0, 1].set_title("Histogram ảnh gốc")
        self.histogram_ax[1, 0].set_title("Hình ảnh kết quả")
        self.histogram_ax[1, 1].set_title("Histogram ảnh kết quả")
        
        # Tạo canvas matplotlib
        self.histogram_canvas = FigureCanvasTkAgg(self.histogram_fig, master=histogram_frame)
        self.histogram_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Nút phân tích
        Button(histogram_frame, text="Phân tích histogram", command=self.show_histogram,
               bg="#009688", fg="white", font=("Arial", 12), padx=10).pack(pady=10)
    
    def update_alpha(self, value):
        self.alpha = float(value)
        self.alpha_value.config(text=f"{self.alpha:.1f}")
    
    def update_k(self, value):
        self.k_iterations = int(float(value))
        self.k_value.config(text=f"{self.k_iterations}")
    
    def update_kernel_size(self, value):
        # Đảm bảo kernel_size luôn là số lẻ
        self.kernel_size = int(float(value))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.kernel_value.config(text=f"{self.kernel_size}")
    
    def update_local_kernel_size(self, value):
        self.local_kernel_size = int(float(value))
        self.local_kernel_value.config(text=f"{self.local_kernel_size}")
    
    def update_s_factor(self, value):
        self.s_factor = float(value)
        self.s_value.config(text=f"{self.s_factor:.1f}")
    
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
            self.display_overview_images(self.original_image, self.original_image)
            
            # Reset kết quả
            self.gaussian_result = None
            self.cluster_result = None
            self.edge_preserve_result = None
            self.intermediate_results = {}
            
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
            self.gaussian_result = enhance_with_gausian_filter(
                self.original_image,
                k=self.k_iterations,
                alpha=self.alpha,
                kernel_size=self.kernel_size
            )
            
            # Hiển thị kết quả
            self.root.after(0, lambda: self.display_overview_images(self.original_image, self.gaussian_result))
            self.root.after(0, lambda: self.status_label.config(text="Đã áp dụng bộ lọc Gaussian thành công!"))
            self.root.after(0, lambda: self.progress.config(value=100))
            
            # Chuyển sang tab tổng quan
            self.root.after(0, lambda: self.notebook.select(0))
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
            
            self.cluster_result = cluster_filter_v3(
                self.original_image, starting_image, k=self.k_iterations, 
                beta=beta, alpha=self.alpha, kernel_size=self.kernel_size
            )
            
            # Hiển thị kết quả
            self.root.after(0, lambda: self.display_overview_images(self.original_image, self.cluster_result))
            self.root.after(0, lambda: self.status_label.config(text="Đã áp dụng bộ lọc phân cụm thành công!"))
            self.root.after(0, lambda: self.progress.config(value=100))
            
            # Chuyển sang tab tổng quan
            self.root.after(0, lambda: self.notebook.select(0))
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
            
            # Bước 1: Lọc ảnh với bộ lọc phân cụm
            self.progress['value'] = 10
            self.status_label.config(text="Bước 1/6: Lọc phân cụm...")
            self.root.update()
            
            starting_image, beta = get_starting_point_v2(
                self.original_image, alpha=self.alpha, kernel_size=self.kernel_size
            )
            
            filtered_image = starting_image.copy()
            for i in range(self.k_iterations):
                filtered_image = cluster_filter_v3(
                    self.original_image, filtered_image, alpha=self.alpha, 
                    beta=beta, k=1, kernel_size=self.kernel_size
                )
                self.progress['value'] = 10 + (i+1)*10/self.k_iterations
                self.root.update()
            
            self.intermediate_results['filtered'] = filtered_image
            
            # Bước 2: Tính hiệu số
            self.progress['value'] = 30
            self.status_label.config(text="Bước 2/6: Tính hiệu số...")
            self.root.update()
            
            Image_i = filtered_image
            Image_d = self.original_image - Image_i
            self.intermediate_results['diff'] = Image_d
            
            # Bước 3: Tính trung bình và phương sai cục bộ
            self.progress['value'] = 40
            self.status_label.config(text="Bước 3/6: Tính trung bình và phương sai cục bộ...")
            self.root.update()
            
            M, V = compute_local_mean_var(Image_d, kernel_size=self.local_kernel_size)
            self.intermediate_results['local_mean'] = M
            self.intermediate_results['local_var'] = V
            
            # Bước 4: Tạo mặt nạ thông qua ngưỡng
            self.progress['value'] = 60
            self.status_label.config(text="Bước 4/6: Tạo mặt nạ...")
            self.root.update()
            
            threshold = V * 2.5
            Image_m = np.where(np.abs(Image_d - M) < threshold, Image_i, self.original_image)
            self.intermediate_results['mask'] = Image_m
            
            # Bước 5: Trừ mặt nạ từ ảnh gốc
            self.progress['value'] = 80
            self.status_label.config(text="Bước 5/6: Tạo kết quả...")
            self.root.update()
            
            Image_o = self.original_image - self.s_factor * Image_m
            
            # Bước 6: Tự động điều chỉnh phạm vi
            self.progress['value'] = 90
            self.status_label.config(text="Bước 6/6: Auto rescale...")
            self.root.update()
            
            # Tính trung bình và độ lệch chuẩn
            m = np.mean(Image_o)
            v = np.std(Image_o)
            
            # Điều chỉnh phạm vi [m - 2.5*v, m + 2.5*v]
            lower = m - 2.5 * v
            upper = m + 2.5 * v
            
            # Chuẩn hóa hình ảnh trong phạm vi [0, 255]
            Image_o = np.clip(Image_o, lower, upper)
            self.edge_preserve_result = ((Image_o - lower) / (upper - lower) * 255).astype(np.uint8)
            self.intermediate_results['result'] = self.edge_preserve_result
            
            # Hiển thị kết quả
            self.root.after(0, lambda: self.display_overview_images(self.original_image, self.edge_preserve_result))
            self.root.after(0, lambda: self.display_steps_images())
            self.root.after(0, lambda: self.status_label.config(text="Đã áp dụng bộ lọc bảo toàn cạnh thành công!"))
            self.root.after(0, lambda: self.progress.config(value=100))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Lỗi: {str(e)}"))
        finally:
            self.processing = False
    
    def compare_filters(self):
        if self.original_image is None:
            self.status_label.config(text="Chưa có hình ảnh nào được tải lên!")
            return
        
        if self.processing:
            self.status_label.config(text="Đang xử lý, vui lòng đợi...")
            return
        
        # Bắt đầu xử lý trong một luồng riêng biệt
        self.processing = True
        threading.Thread(target=self._compare_filters_thread).start()
    
    def _compare_filters_thread(self):
        try:
            self.status_label.config(text="Đang so sánh các bộ lọc...")
            self.progress['value'] = 0
            self.root.update()
            
            # Nếu chưa có kết quả, thực hiện các bộ lọc
            if self.gaussian_result is None:
                self.status_label.config(text="Đang áp dụng bộ lọc Gaussian...")
                self.progress['value'] = 10
                self.root.update()
                
                self.gaussian_result = enhance_with_gausian_filter(
                    self.original_image,
                    k=self.k_iterations,
                    alpha=self.alpha,
                    kernel_size=self.kernel_size
                )
            
            if self.cluster_result is None:
                self.status_label.config(text="Đang áp dụng bộ lọc phân cụm...")
                self.progress['value'] = 40
                self.root.update()
                
                starting_image, beta = get_starting_point_v2(
                    self.original_image, alpha=self.alpha, kernel_size=self.kernel_size
                )
                
                self.cluster_result = cluster_filter_v3(
                    self.original_image, starting_image, k=self.k_iterations, 
                    beta=beta, alpha=self.alpha, kernel_size=self.kernel_size
                )
            
            if self.edge_preserve_result is None:
                self.status_label.config(text="Đang áp dụng bộ lọc bảo toàn cạnh...")
                self.progress['value'] = 70
                self.root.update()
                
                self.edge_preserve_result = enhance_with_cluster_filter(
                    self.original_image,
                    k=self.k_iterations,
                    alpha=self.alpha,
                    kernel_size=self.kernel_size
                )
            
            # Hiển thị kết quả so sánh
            self.root.after(0, lambda: self.display_comparison_images())
            self.root.after(0, lambda: self.status_label.config(text="Đã so sánh các bộ lọc thành công!"))
            self.root.after(0, lambda: self.progress.config(value=100))
            
            # Chuyển sang tab so sánh
            self.root.after(0, lambda: self.notebook.select(2))
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Lỗi: {str(e)}"))
        finally:
            self.processing = False
    
    def show_intermediate_steps(self):
        if not self.intermediate_results:
            messagebox.showwarning("Chưa có kết quả", "Vui lòng áp dụng bộ lọc bảo toàn cạnh trước!")
            return
        
        # Hiển thị các bước trung gian
        self.display_steps_images()
        
        # Chuyển sang tab "Các bước trung gian"
        self.notebook.select(1)
    
    def show_histogram(self):
        if self.original_image is None:
            messagebox.showwarning("Chưa có hình ảnh", "Vui lòng tải hình ảnh trước!")
            return
        
        if self.edge_preserve_result is None:
            messagebox.showwarning("Chưa có kết quả", "Vui lòng áp dụng bộ lọc bảo toàn cạnh trước!")
            return
        
        # Xóa nội dung trục hiện tại
        for ax in self.histogram_ax.flat:
            ax.clear()
        
        # Hiển thị hình ảnh gốc
        self.histogram_ax[0, 0].imshow(self.original_image, cmap='gray')
        self.histogram_ax[0, 0].set_title("Hình ảnh gốc")
        self.histogram_ax[0, 0].axis('off')
        
        # Hiển thị histogram ảnh gốc
        self.histogram_ax[0, 1].hist(self.original_image.flatten(), bins=256, range=(0, 256), color='blue', alpha=0.7)
        self.histogram_ax[0, 1].set_title("Histogram ảnh gốc")
        self.histogram_ax[0, 1].set_xlabel("Giá trị pixel")
        self.histogram_ax[0, 1].set_ylabel("Số lượng")
        
        # Hiển thị hình ảnh kết quả
        self.histogram_ax[1, 0].imshow(self.edge_preserve_result, cmap='gray')
        self.histogram_ax[1, 0].set_title("Hình ảnh kết quả")
        self.histogram_ax[1, 0].axis('off')
        
        # Hiển thị histogram ảnh kết quả
        self.histogram_ax[1, 1].hist(self.edge_preserve_result.flatten(), bins=256, range=(0, 256), color='red', alpha=0.7)
        self.histogram_ax[1, 1].set_title("Histogram ảnh kết quả")
        self.histogram_ax[1, 1].set_xlabel("Giá trị pixel")
        self.histogram_ax[1, 1].set_ylabel("Số lượng")
        
        # Cập nhật canvas
        self.histogram_fig.tight_layout()
        self.histogram_canvas.draw()
        
        # Chuyển sang tab "Phân tích histogram"
        self.notebook.select(3)
    
    def display_overview_images(self, img1, img2):
        # Xóa nội dung trục hiện tại
        self.overview_ax[0].clear()
        self.overview_ax[1].clear()
        
        # Hiển thị hình ảnh gốc
        self.overview_ax[0].imshow(img1, cmap='gray')
        self.overview_ax[0].set_title("Hình ảnh gốc")
        self.overview_ax[0].axis('off')
        
        # Hiển thị hình ảnh kết quả
        self.overview_ax[1].imshow(img2, cmap='gray')
        self.overview_ax[1].set_title("Kết quả")
        self.overview_ax[1].axis('off')
        
        # Cập nhật canvas
        self.overview_fig.tight_layout()
        self.overview_canvas.draw()
    
    def display_steps_images(self):
        if not self.intermediate_results:
            return
        
        # Xóa nội dung trục hiện tại
        for ax in self.steps_axes.values():
            ax.clear()
        
        # Hiển thị hình ảnh gốc
        self.steps_axes['original'].imshow(self.original_image, cmap='gray')
        self.steps_axes['original'].set_title("Hình ảnh gốc")
        self.steps_axes['original'].axis('off')
        
        # Hiển thị hình ảnh sau lọc phân cụm
        if 'filtered' in self.intermediate_results:
            self.steps_axes['filtered'].imshow(self.intermediate_results['filtered'], cmap='gray')
            self.steps_axes['filtered'].set_title("Bước 1: Lọc phân cụm")
            self.steps_axes['filtered'].axis('off')
        
        # Hiển thị hình ảnh hiệu số
        if 'diff' in self.intermediate_results:
            self.steps_axes['diff'].imshow(self.intermediate_results['diff'], cmap='gray')
            self.steps_axes['diff'].set_title("Bước 2: Hiệu số")
            self.steps_axes['diff'].axis('off')
        
        # Hiển thị hình ảnh trung bình cục bộ
        if 'local_mean' in self.intermediate_results:
            self.steps_axes['local_mean'].imshow(self.intermediate_results['local_mean'], cmap='gray')
            self.steps_axes['local_mean'].set_title("Bước 3: Trung bình cục bộ")
            self.steps_axes['local_mean'].axis('off')
        
        # Hiển thị hình ảnh mặt nạ
        if 'mask' in self.intermediate_results:
            self.steps_axes['mask'].imshow(self.intermediate_results['mask'], cmap='gray')
            self.steps_axes['mask'].set_title("Bước 4: Mặt nạ")
            self.steps_axes['mask'].axis('off')
        
        # Hiển thị hình ảnh kết quả
        if 'result' in self.intermediate_results:
            self.steps_axes['result'].imshow(self.intermediate_results['result'], cmap='gray')
            self.steps_axes['result'].set_title("Bước 5+6: Kết quả")
            self.steps_axes['result'].axis('off')
        
        # Cập nhật canvas
        self.steps_fig.tight_layout()
        self.steps_canvas.draw()
    
    def display_comparison_images(self):
        # Xóa nội dung trục hiện tại
        for ax in self.comparison_ax.flat:
            ax.clear()
        
        # Hiển thị hình ảnh gốc
        self.comparison_ax[0, 0].imshow(self.original_image, cmap='gray')
        self.comparison_ax[0, 0].set_title("Hình ảnh gốc")
        self.comparison_ax[0, 0].axis('off')
        plot_img(self.comparison_ax[0, 0], self.original_image, "Hình ảnh gốc", cmap=plt.get_cmap("gray"))

        # Hiển thị kết quả bộ lọc Gaussian
        if self.gaussian_result is not None:
            self.comparison_ax[0, 1].imshow(self.gaussian_result, cmap='gray')
            self.comparison_ax[0, 1].set_title("Bộ lọc Gaussian")
            self.comparison_ax[0, 1].axis('off')
        plot_img(self.comparison_ax[0, 1], self.gaussian_result, "Bộ lọc Gaussian", cmap=plt.get_cmap("gray"), vmin=None, vmax=None)
        
        # Hiển thị kết quả bộ lọc phân cụm
        if self.cluster_result is not None:
            self.comparison_ax[1, 0].imshow(self.cluster_result, cmap='gray')
            self.comparison_ax[1, 0].set_title("Bộ lọc phân cụm")
            self.comparison_ax[1, 0].axis('off')
        plot_img(self.comparison_ax[1, 0], self.cluster_result, "Bộ lọc phân cụm", cmap=plt.get_cmap("gray"), vmin=None, vmax=None)
        
        # Hiển thị kết quả bộ lọc bảo toàn cạnh
        if self.edge_preserve_result is not None:
            self.comparison_ax[1, 1].imshow(self.edge_preserve_result, cmap='gray')
            self.comparison_ax[1, 1].set_title("Bộ lọc bảo toàn cạnh")
            self.comparison_ax[1, 1].axis('off')
        plot_img(self.comparison_ax[1, 1], self.edge_preserve_result, "Bộ lọc bảo toàn cạnh", cmap=plt.get_cmap("gray"), vmin=None, vmax=None)
        
        # Cập nhật canvas
        self.comparison_fig.tight_layout()
        self.comparison_canvas.draw()
    
    def save_image(self):
        if self.edge_preserve_result is None:
            messagebox.showwarning("Chưa có kết quả", "Vui lòng áp dụng bộ lọc trước khi lưu!")
            return
        
        # Hiển thị hộp thoại lưu file
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Lưu hình ảnh
                cv2.imwrite(file_path, self.edge_preserve_result)
                self.status_label.config(text=f"Đã lưu hình ảnh tại: {file_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu hình ảnh: {str(e)}")
    
    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("Hướng dẫn sử dụng")
        help_window.geometry("600x700")
        self.center_window(help_window)
        
        # Tạo frame chứa nội dung
        content_frame = Frame(help_window, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Tiêu đề
        Label(content_frame, text="HƯỚNG DẪN SỬ DỤNG BỘ LỌC BẢO TOÀN CẠNH", 
              font=("Arial", 14, "bold")).pack(pady=10)
        
        # Tạo widget Text để hiển thị hướng dẫn
        help_text = tk.Text(content_frame, wrap=tk.WORD, width=80, height=30)
        help_text.pack(fill=tk.BOTH, expand=True)
        
        # Thêm thanh cuộn
        scrollbar = tk.Scrollbar(help_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        help_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=help_text.yview)
        
        # Nội dung hướng dẫn
        help_content = """
Phần mềm Bộ lọc bảo toàn cạnh - Phiên bản nâng cao

1. Giới thiệu
   Phần mềm này triển khai bộ lọc bảo toàn cạnh dựa trên bài báo "Image Enhancement by Edge-Preserving Filtering" của Yiu-fai Wong. Bộ lọc này có khả năng tăng cường chất lượng hình ảnh bằng cách:
   - Loại bỏ nhiễu xung
   - Làm mịn nhiễu không xung
   - Bảo toàn các cạnh quan trọng

2. Các tính năng chính
   - Tải và xử lý hình ảnh grayscale
   - Áp dụng ba loại bộ lọc: Gaussian, phân cụm, và bảo toàn cạnh
   - So sánh hiệu quả giữa các bộ lọc
   - Hiển thị các bước trung gian trong quá trình xử lý
   - Phân tích histogram trước và sau khi xử lý
   - Lưu kết quả ra tập tin

3. Hướng dẫn sử dụng
   3.1. Tải hình ảnh
      - Nhấp vào nút "Tải hình ảnh" để mở hộp thoại chọn tệp
      - Chọn một tệp hình ảnh grayscale hoặc màu (sẽ được chuyển đổi thành grayscale)

   3.2. Điều chỉnh tham số
      - Alpha: Điều khiển quy mô trong không gian đầu vào
      - Số lần lặp (k): Số lần áp dụng bộ lọc phân cụm
      - Kích thước kernel: Kích thước cửa sổ lọc
      - Kích thước kernel cục bộ: Kích thước cửa sổ tính trung bình và phương sai cục bộ
      - Hệ số s: Hệ số trong công thức kết quả cuối cùng

   3.3. Áp dụng bộ lọc
      - Nhấp vào nút "Áp dụng bộ lọc bảo toàn cạnh" để xử lý hình ảnh
      - Sử dụng menu "Bộ lọc" để áp dụng các bộ lọc khác nhau

   3.4. Xem kết quả
      - Tab "Tổng quan": Hiển thị hình ảnh gốc và kết quả
      - Tab "Các bước trung gian": Hiển thị các bước trong quá trình xử lý
      - Tab "So sánh các bộ lọc": So sánh hiệu quả giữa các bộ lọc khác nhau
      - Tab "Phân tích histogram": Hiển thị histogram trước và sau khi xử lý

   3.5. Lưu kết quả
      - Nhấp vào nút "Lưu kết quả" để lưu hình ảnh đã xử lý ra tập tin

4. Các tham số và ý nghĩa
   - Alpha (α): Điều khiển mức độ ảnh hưởng của các điểm lân cận. Giá trị càng nhỏ thì vùng ảnh hưởng càng lớn.
   - Số lần lặp (k): Số lần áp dụng bộ lọc phân cụm. Giá trị càng lớn thì hiệu ứng làm mịn càng mạnh.
   - Kích thước kernel: Kích thước cửa sổ lọc. Giá trị càng lớn thì hiệu ứng làm mịn càng mạnh.
   - Kích thước kernel cục bộ: Kích thước cửa sổ tính trung bình và phương sai cục bộ. Giá trị càng lớn thì vùng xác định ngưỡng càng rộng.
   - Hệ số s: Hệ số trong công thức kết quả cuối cùng (Image_o = image - s * Image_m). Giá trị càng lớn thì hiệu ứng tăng cường cạnh càng mạnh.

5. Lưu ý
   - Quá trình xử lý có thể mất thời gian tùy thuộc vào kích thước hình ảnh và tham số.
   - Với hình ảnh lớn, nên chọn tham số phù hợp để tránh quá trình xử lý kéo dài.
   - Kết quả tốt nhất thường đạt được khi alpha trong khoảng 0.2-0.5 và k trong khoảng 3-7.
        """
        
        help_text.insert(tk.END, help_content)
        help_text.config(state=tk.DISABLED)  # Chỉ đọc
        
        # Nút đóng
        Button(content_frame, text="Đóng", command=help_window.destroy,
               bg="#f44336", fg="white", font=("Arial", 12), padx=10).pack(pady=10)
    
    def show_about(self):
        about_window = tk.Toplevel(self.root)
        about_window.title("Thông tin")
        about_window.geometry("400x300")
        self.center_window(about_window)
        
        # Tạo frame chứa nội dung
        content_frame = Frame(about_window, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Tiêu đề
        Label(content_frame, text="BỘ LỌC BẢO TOÀN CẠNH", 
              font=("Arial", 14, "bold")).pack(pady=10)
        
        # Nội dung
        Label(content_frame, text="Phiên bản: 1.0.0", font=("Arial", 12)).pack(pady=5)
        Label(content_frame, text="Dựa trên bài báo:", font=("Arial", 12)).pack(pady=5)
        Label(content_frame, text='"Image Enhancement by Edge-Preserving Filtering"', 
              font=("Arial", 12, "italic")).pack()
        Label(content_frame, text="Tác giả: Yiu-fai Wong", font=("Arial", 12)).pack(pady=5)
        Label(content_frame, text="IEEE International Conference on Image Processing, 1994", 
              font=("Arial", 12)).pack()
        
        # Nút đóng
        Button(content_frame, text="Đóng", command=about_window.destroy,
               bg="#f44336", fg="white", font=("Arial", 12), padx=10).pack(pady=20)
    
    def on_closing(self):
        if messagebox.askokcancel("Thoát", "Bạn có chắc chắn muốn thoát không?"):
            plt.close('all')  # Đóng tất cả các hình vẽ matplotlib
            self.root.destroy()

# Khởi chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedFilterDemo(root)
    
    # Đặt cửa sổ chính ở giữa màn hình sau khi hoàn tất khởi tạo
    root.update_idletasks()
    app.center_window(root)
    
    root.mainloop()