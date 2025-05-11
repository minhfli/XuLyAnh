import tkinter as tk
import sys
import os
from tkinter import messagebox

# Thêm thư mục hiện tại vào đường dẫn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class DemoLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Bộ lọc bảo toàn cạnh - Launcher")
        self.root.configure(bg='#f0f0f0')
        
        # Tạo tiêu đề
        tk.Label(
            root, 
            text="Chọn demo để khởi chạy", 
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            pady=20
        ).pack()
        
        # Tạo nút chạy demo cơ bản
        tk.Button(
            root,
            text="Demo cơ bản",
            command=self.run_basic_demo,
            font=("Arial", 12),
            bg="#4CAF50",
            fg="white",
            width=20,
            height=2,
            relief=tk.RAISED,
            bd=3
        ).pack(pady=10)
        
        # Tạo nút chạy demo nâng cao
        tk.Button(
            root,
            text="Demo nâng cao",
            command=self.run_advanced_demo,
            font=("Arial", 12),
            bg="#2196F3",
            fg="white",
            width=20,
            height=2,
            relief=tk.RAISED,
            bd=3
        ).pack(pady=10)
        
        # Tạo nút thoát
        tk.Button(
            root,
            text="Thoát",
            command=root.quit,
            font=("Arial", 12),
            bg="#f44336",
            fg="white",
            width=20,
            height=1
        ).pack(pady=10)
    
    def run_basic_demo(self):
        try:
            from demo import EdgePreservingFilterDemo
            self.root.withdraw()  # Ẩn cửa sổ launcher
            new_root = tk.Toplevel()
            app = EdgePreservingFilterDemo(new_root)
            new_root.protocol("WM_DELETE_WINDOW", lambda: self.on_demo_close(new_root))
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể khởi chạy demo cơ bản: {str(e)}")
    
    def run_advanced_demo(self):
        try:
            from advanced_demo import AdvancedFilterDemo
            self.root.withdraw()  # Ẩn cửa sổ launcher
            new_root = tk.Toplevel()
            app = AdvancedFilterDemo(new_root)
            new_root.protocol("WM_DELETE_WINDOW", lambda: self.on_demo_close(new_root))
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể khởi chạy demo nâng cao: {str(e)}")
    
    def on_demo_close(self, window):
        window.destroy()
        self.root.deiconify()  # Hiện lại cửa sổ launcher

if __name__ == "__main__":
    # Kiểm tra các thư viện cần thiết
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageTk
        import matplotlib.pyplot as plt
    except ImportError as e:
        module_name = str(e).split("'")[1]
        messagebox.showerror(
            "Lỗi thư viện",
            f"Thiếu thư viện {module_name}. Hãy cài đặt bằng lệnh:\npip install {module_name}"
        )
        sys.exit(1)
    
    # Khởi chạy launcher
    root = tk.Tk()
    app = DemoLauncher(root)
    root.mainloop()