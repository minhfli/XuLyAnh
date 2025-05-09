import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PIL import Image, ImageTk
import time
import threading

# Thêm đường dẫn hiện tại vào sys.path để có thể import module filter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.filter import get_starting_point_v2, cluster_filter_v3, compute_local_mean_var

class AutoDemo:
    def __init__(self, root, on_close_callback=None):
        self.root = root
        self.on_close_callback = on_close_callback
        self.root.title("Bộ lọc bảo toàn cạnh - Demo tự động")
        self.root.geometry("1280x800")
        
        # Chuẩn bị thư mục hình ảnh mẫu
        self.assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
        self.sample_images = [
            os.path.join(self.assets_dir, f) for f in os.listdir(self.assets_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]
        
        if not self.sample_images:
            tk.Label(
                self.root,
                text="Không tìm thấy hình ảnh mẫu trong thư mục assets!",
                font=("Arial", 14),
                fg="red"
            ).pack(pady=20)
            return
        
        # Biến điều khiển
        self.current_image_index = 0
        self.is_running = False
        self.pause_event = threading.Event()
        self.pause_event.set()  # Bắt đầu ở trạng thái không tạm dừng
        
        # Tạo giao diện
        self.create_ui()
        
        # Bắt đầu demo
        self.start_demo()
    
    def create_ui(self):
        # Tạo frame điều khiển
        control_frame = tk.Frame(self.root, bg='#f0f0f0')
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Tạo nút điều khiển
        self.play_pause_button = tk.Button(
            control_frame,
            text="Tạm dừng",
            command=self.toggle_pause,
            font=("Arial", 12),
            bg="#2196F3",
            fg="white",
            width=10
        )
        self.play_pause_button.pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            control_frame,
            text="Hình ảnh trước",
            command=self.prev_image,
            font=("Arial", 12),
            bg="#4CAF50",
            fg="white",
            width=15
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            control_frame,
            text="Hình ảnh tiếp",
            command=self.next_image,
            font=("Arial", 12),
            bg="#4CAF50",
            fg="white",
            width=15
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            control_frame,
            text="Thoát",
            command=self.close_demo,
            font=("Arial", 12),
            bg="#f44336",
            fg="white",
            width=10
        ).pack(side=tk.RIGHT, padx=10)
        
        # Tạo trạng thái
        self.status_label = tk.Label(
            control_frame,
            text="Đang xử lý...",
            font=("Arial", 12),
            bg='#f0f0f0'
        )
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Tạo frame hiển thị
        display_frame = tk.Frame(self.root, bg='#e0e0e0')
        display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tạo figure để hiển thị kết quả
        self.fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
        
        self.axes = {
            'original': plt.subplot(gs[0, 0]),
            'gaussian': plt.subplot(gs[0, 1]),
            'cluster': plt.subplot(gs[0, 2]),
            'diff': plt.subplot(gs[1, 0]),
            'mask': plt.subplot(gs[1, 1]),
            'result': plt.subplot(gs[1, 2])
        }
        
        # Tạo canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Thiết lập đóng cửa sổ
        self.root.protocol("WM_DELETE_WINDOW", self.close_demo)
    
    def toggle_pause(self):
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.play_pause_button.config(text="Tiếp tục")
        else:
            self.pause_event.set()
            self.play_pause_button.config(text="Tạm dừng")
    
    def prev_image(self):
        self.current_image_index = (self.current_image_index - 1) % len(self.sample_images)
        self.process_current_image()
    
    def next_image(self):
        self.current_image_index = (self.current_image_index + 1) % len(self.sample_images)
        self.process_current_image()
    
    def start_demo(self):
        self.is_running = True
        threading.Thread(target=self.demo_thread, daemon=True).start()
    
    def demo_thread(self):
        while self.is_running:
            self.pause_event.wait()  # Chờ nếu đang tạm dừng
            
            if not self.is_running:
                break
            
            # Xử lý hình ảnh hiện tại
            self.process_current_image()
            
            # Chờ 5 giây trước khi chuyển sang hình ảnh tiếp theo
            for _ in range(50):  # 5 seconds with 0.1s intervals
                if not self.is_running:
                    break
                self.pause_event.wait()  # Chờ nếu đang tạm dừng
                time.sleep(0.1)
            
            if self.is_running and self.pause_event.is_set():
                # Chuyển sang hình ảnh tiếp theo
                self.current_image_index = (self.current_image_index + 1) % len(self.sample_images)
    
    def process_current_image(self):
        if not self.is_running:
            return
        
        # Hiển thị đang xử lý
        self.root.after(0, lambda: self.status_label.config(
            text=f"Đang xử lý {os.path.basename(self.sample_images[self.current_image_index])}..."
        ))
        
        try:
            # Đọc hình ảnh
            image_path = self.sample_images[self.current_image_index]
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Xử lý hình ảnh
            alpha = 0.5
            k_iterations = 5
            kernel_size = 11
            
            # Áp dụng bộ lọc Gaussian
            gaussian_result = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
            # Áp dụng bộ lọc phân cụm
            starting_image, beta = get_starting_point_v2(image, alpha, kernel_size)
            cluster_result = cluster_filter_v3(
                image, starting_image, k=k_iterations, beta=beta, alpha=alpha, kernel_size=kernel_size
            )
            
            # Tính hiệu số
            diff_image = image - cluster_result
            
            # Tính trung bình và phương sai cục bộ
            M, V = compute_local_mean_var(diff_image)
            
            # Tạo mặt nạ
            threshold = V * 2.5
            mask_image = np.where(np.abs(diff_image - M) < threshold, cluster_result, image)
            
            # Tạo kết quả cuối cùng
            s = 0.5
            final_result = image - s * mask_image
            
            # Hiển thị kết quả
            self.display_results(
                image, gaussian_result, cluster_result, 
                diff_image, mask_image, final_result
            )
            
            # Cập nhật trạng thái
            self.root.after(0, lambda: self.status_label.config(
                text=f"Đã xử lý: {os.path.basename(image_path)}"
            ))
        
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(
                text=f"Lỗi: {str(e)}"
            ))
    
    def display_results(self, original, gaussian, cluster, diff, mask, result):
        # Xóa các trục
        for ax in self.axes.values():
            ax.clear()
        
        # Hiển thị hình ảnh
        self.axes['original'].imshow(original, cmap='gray')
        self.axes['original'].set_title('Hình ảnh gốc')
        self.axes['original'].axis('off')
        
        self.axes['gaussian'].imshow(gaussian, cmap='gray')
        self.axes['gaussian'].set_title('Bộ lọc Gaussian')
        self.axes['gaussian'].axis('off')
        
        self.axes['cluster'].imshow(cluster, cmap='gray')
        self.axes['cluster'].set_title('Bộ lọc phân cụm')
        self.axes['cluster'].axis('off')
        
        self.axes['diff'].imshow(diff, cmap='gray')
        self.axes['diff'].set_title('Hiệu số')
        self.axes['diff'].axis('off')
        
        self.axes['mask'].imshow(mask, cmap='gray')
        self.axes['mask'].set_title('Mặt nạ')
        self.axes['mask'].axis('off')
        
        self.axes['result'].imshow(result, cmap='gray')
        self.axes['result'].set_title('Kết quả')
        self.axes['result'].axis('off')
        
        # Cập nhật canvas
        self.fig.tight_layout()
        self.root.after(0, self.canvas.draw)
    
    def close_demo(self):
        self.is_running = False
        self.root.destroy()
        
        if self.on_close_callback:
            self.on_close_callback()

def run_auto_demo(on_close_callback=None):
    root = tk.Tk()
    app = AutoDemo(root, on_close_callback)
    root.mainloop()

if __name__ == "__main__":
    run_auto_demo()