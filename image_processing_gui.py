import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QSlider, QPushButton, QComboBox,
    QCheckBox, QFileDialog, QHBoxLayout, QSpinBox, QDoubleSpinBox, QProgressBar, QFrame, QScrollArea, QGridLayout, QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
import importlib.util
import subprocess

class ProcessThread(QThread):
    progress_signal = pyqtSignal(str, int)  # 用于更新进度
    image_paths_signal = pyqtSignal(str)   # 传递生成的图片路径

    def __init__(self, module_name, params, run_count):
        super().__init__()
        self.module_name = module_name
        self.params = params
        self.run_count = run_count  # 新增 run_count 参数

    def run(self):
        # 动态加载模块
        spec = importlib.util.spec_from_file_location(self.module_name, f"./{self.module_name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        try:
            for i in range(self.run_count):
                output_filename = f"output/result_{i + 1}.png"
                self.params['output_name'] = output_filename
                final_image_path = output_filename

                # 调用处理函数
                module.process_image_with_novelai(**self.params)

                # 确保文件生成后发送信号
                if os.path.exists(final_image_path) and os.path.getsize(final_image_path) > 0:
                    self.image_paths_signal.emit(final_image_path)
                else:
                    self.progress_signal.emit(f"Error: {final_image_path} generation failed.", 100)
                    continue

                # 更新进度条
                progress = int((i + 1) / self.run_count * 100)
                self.progress_signal.emit(f"Processed image {i + 1}/{self.run_count}", progress)

        except Exception as e:
            self.progress_signal.emit(f"Error: {str(e)}", 100)



class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.result_images = [] 
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("NovelAI Image Processor - by Komachan")
        self.setGeometry(100, 100, 1200, 700)

        main_layout = QHBoxLayout()

        center_layout = QVBoxLayout()
        self.input_image_label = QLabel("Input Image")
        self.input_image_label.setAlignment(Qt.AlignCenter)
        self.input_image_label.setFixedSize(600, 600)
        self.input_image_label.setFrameShape(QFrame.Box)
        center_layout.addWidget(self.input_image_label)

        # Input file section
        self.file_button = QPushButton("Select Input Image")
        self.file_button.clicked.connect(self.select_file)
        center_layout.addWidget(self.file_button)

        # 运行次数设置
        left_layout = QVBoxLayout()
        self.run_count_label = QLabel("Run Count:")
        self.run_count_input = QSpinBox()
        self.run_count_input.setRange(1, 100)
        self.run_count_input.setValue(1)
        left_layout.addWidget(self.run_count_label)
        left_layout.addWidget(self.run_count_input)

        # 保存结果按钮
        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        left_layout.addWidget(self.save_button)

        # 清空结果按钮
        self.clear_button = QPushButton("Clear Results")
        self.clear_button.clicked.connect(self.clear_results)
        left_layout.addWidget(self.clear_button)

        # 导出设置
        self.export_button = QPushButton("Export Settings")
        self.export_button.clicked.connect(self.export_settings)
        left_layout.addWidget(self.export_button)

        # 导入设置
        self.import_button = QPushButton("Import Settings")
        self.import_button.clicked.connect(self.import_settings)
        left_layout.addWidget(self.import_button)
        
        self.file_label = QLabel("Input Image: Not selected")
        left_layout.addWidget(self.file_label)

        # Functionality selection
        self.function_label = QLabel("Select Functionality:")
        left_layout.addWidget(self.function_label)

        self.function_combo = QComboBox()
        self.function_combo.addItems(["1-2.5倍", "5-5倍", "5-5倍BZ"])
        left_layout.addWidget(self.function_combo)

        # Positive prompt
        self.positive_prompt_label = QLabel("Positive Prompt:")
        left_layout.addWidget(self.positive_prompt_label)
        self.positive_prompt_input = QTextEdit()
        self.positive_prompt_input.setPlaceholderText("Enter Positive Prompt")
        self.positive_prompt_input.setWordWrapMode(True)
        left_layout.addWidget(self.positive_prompt_input)

        # Negative prompt
        self.negative_prompt_label = QLabel("Negative Prompt (default provided):")
        left_layout.addWidget(self.negative_prompt_label)
        self.negative_prompt_input = QTextEdit(
            "nsfw, lowres, {bad}, error, fewer, extra, missing, "
            "worst quality, jpeg artifacts, bad quality, watermark, "
            "unfinished, displeasing, chromatic aberration, signature, "
            "extra digits, artistic error, username, scan, [abstract]"
        )
        self.negative_prompt_input.setPlaceholderText("Enter Negative Prompt")
        self.negative_prompt_input.setWordWrapMode(True)
        left_layout.addWidget(self.negative_prompt_input)

        # Scale
        self.scale_label = QLabel("Scale (0-10):")
        left_layout.addWidget(self.scale_label)
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(0, 10)
        self.scale_slider.setValue(5)
        left_layout.addWidget(self.scale_slider)

        # Sampler
        self.sampler_label = QLabel("Sampler:")
        left_layout.addWidget(self.sampler_label)
        self.sampler_combo = QComboBox()
        self.sampler_combo.addItems(["k_dpmpp_2s_ancestral", "k_dpmpp_2m_sde"])  # 可以扩展选项
        left_layout.addWidget(self.sampler_combo)

        # Steps
        self.steps_label = QLabel("Steps (1-28):")
        left_layout.addWidget(self.steps_label)
        self.steps_input = QSpinBox()
        self.steps_input.setRange(1, 28)
        self.steps_input.setValue(28)
        left_layout.addWidget(self.steps_input)

        # Strength
        self.strength_label = QLabel("Strength (0.0-0.99):")
        left_layout.addWidget(self.strength_label)
        self.strength_input = QDoubleSpinBox()
        self.strength_input.setRange(0.0, 0.99)
        self.strength_input.setValue(0.3)
        self.strength_input.setSingleStep(0.01)
        left_layout.addWidget(self.strength_input)

        # Noise
        self.noise_label = QLabel("Noise (0.0-0.99):")
        left_layout.addWidget(self.noise_label)
        self.noise_input = QDoubleSpinBox()
        self.noise_input.setRange(0.0, 0.99)
        self.noise_input.setValue(0.0)
        self.noise_input.setSingleStep(0.01)
        left_layout.addWidget(self.noise_input)

        # Noise schedule
        self.noise_schedule_label = QLabel("Noise Schedule:")
        left_layout.addWidget(self.noise_schedule_label)
        self.noise_schedule_combo = QComboBox()
        self.noise_schedule_combo.addItems(["karras", "native"])  # 可以扩展选项
        left_layout.addWidget(self.noise_schedule_combo)

        # API Key
        self.api_key_label = QLabel("API Key:")
        left_layout.addWidget(self.api_key_label)
        self.api_key_input = QLineEdit()
        left_layout.addWidget(self.api_key_input)

        # Use WD
        self.use_wd_checkbox = QCheckBox("Use WD Tagger")
        self.use_wd_checkbox.stateChanged.connect(self.toggle_wd_inputs)
        left_layout.addWidget(self.use_wd_checkbox)

        # WD Thresholds
        self.general_thresh_label = QLabel("General Threshold (0.0-1.0):")
        left_layout.addWidget(self.general_thresh_label)
        self.general_thresh_input = QDoubleSpinBox()
        self.general_thresh_input.setRange(0.0, 1.0)
        self.general_thresh_input.setValue(0.35)
        self.general_thresh_input.setSingleStep(0.01)
        self.general_thresh_input.setEnabled(False)
        left_layout.addWidget(self.general_thresh_input)

        self.character_thresh_label = QLabel("Character Threshold (0.0-1.0):")
        left_layout.addWidget(self.character_thresh_label)
        self.character_thresh_input = QDoubleSpinBox()
        self.character_thresh_input.setRange(0.0, 1.0)
        self.character_thresh_input.setValue(0.9)
        self.character_thresh_input.setSingleStep(0.01)
        self.character_thresh_input.setEnabled(False)
        left_layout.addWidget(self.character_thresh_input)

        # Progress bar
        self.progress_label = QLabel("Progress:")
        left_layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)

        # Submit Button
        self.submit_button = QPushButton("Process Image")
        self.submit_button.clicked.connect(self.process_image)
        left_layout.addWidget(self.submit_button)

        # 右侧布局：显示 Output Image 和预览栏
        right_layout = QVBoxLayout()

        # 显示 Output Image
        self.output_image_label = QLabel("Output Image")
        self.output_image_label.setAlignment(Qt.AlignCenter)
        self.output_image_label.setFixedSize(600, 600)
        self.output_image_label.setFrameShape(QFrame.Box)
        right_layout.addWidget(self.output_image_label)

        # Output 预览栏
        self.preview_scroll_area = QScrollArea()
        self.preview_scroll_area.setWidgetResizable(True)
        self.preview_content = QWidget()
        self.preview_layout = QHBoxLayout(self.preview_content)
        self.preview_scroll_area.setWidget(self.preview_content)
        self.preview_scroll_area.setFixedHeight(120)
        right_layout.addWidget(self.preview_scroll_area)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(center_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def add_to_preview(self, image_path):
        """在预览栏中添加缩略图"""
        thumbnail_label = QLabel()
        thumbnail_label.setFixedSize(100, 100)
        thumbnail_label.setFrameShape(QFrame.Box)
        thumbnail_label.setPixmap(QPixmap(image_path).scaled(100, 100, Qt.KeepAspectRatio))
        thumbnail_label.mousePressEvent = lambda event, index=len(self.result_images) - 1: self.show_output_image(index)
        self.preview_layout.addWidget(thumbnail_label)

    def show_output_image(self, index):
        """显示指定索引的输出图片"""
        if 0 <= index < len(self.result_images):
            self.current_output_index = index
            pixmap = QPixmap(self.result_images[index]).scaled(600, 600, Qt.KeepAspectRatio)
            self.output_image_label.setPixmap(pixmap)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path and os.path.exists(file_path):
            self.file_label.setText(f"Input Image: {file_path}")
            self.input_image_path = file_path
            pixmap = QPixmap(file_path).scaled(600, 600, Qt.KeepAspectRatio)
            self.input_image_label.setPixmap(pixmap)
        else:
            self.file_label.setText("Input Image: Not selected")
            self.input_image_label.clear()

    def toggle_wd_inputs(self, state):
        enabled = state == Qt.Checked
        self.general_thresh_input.setEnabled(enabled)
        self.character_thresh_input.setEnabled(enabled)

    def process_image(self):
        self.result_images = []  # 每次处理前清空旧结果
        run_count = self.run_count_input.value()

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        for run in range(run_count):
            module_name = {
                "1-2.5倍": "img2imgWD",
                "5-5倍": "img2imgWD_5",
                "5-5倍BZ": "img2imgWD_BZ",
            }[self.function_combo.currentText()]

            output_name = os.path.join(output_dir, f"result_{run + 1}.png")

            params = {
                "image_path": self.file_label.text().split(": ")[1],
                "positive_prompt": self.positive_prompt_input.toPlainText(),
                "negative_prompt": self.negative_prompt_input.toPlainText(),
                "scale": self.scale_slider.value(),
                "sampler": self.sampler_combo.currentText(),
                "steps": self.steps_input.value(),
                "strength": self.strength_input.value(),
                "noise": self.noise_input.value(),
                "noise_schedule": self.noise_schedule_combo.currentText(),
                "api_key": self.api_key_input.text(),
                "general_thresh": self.general_thresh_input.value(),
                "character_thresh": self.character_thresh_input.value(),
                "use_wd": self.use_wd_checkbox.isChecked(),
                # "run_count": run_count,
                "output_name": output_name,
            }

            # 添加运行编号到日志中
            self.thread = ProcessThread(module_name, params, run_count)
            self.thread.image_paths_signal.connect(self.handle_generated_image)
            self.thread.progress_signal.connect(self.update_progress)
            self.thread.start()
            
    def handle_generated_image(self, image_path):
        if os.path.exists(image_path):
            self.result_images.append(image_path)
            self.add_to_preview(image_path)

            # 如果是第一张图片，显示在大图区域
            if len(self.result_images) == 1:
                self.show_full_output(image_path)
        else:
            print(f"Error: {image_path} does not exist.")

    def display_output(self, image_path):
        if not os.path.exists(image_path):
            print(f"Error: {image_path} does not exist.")  # 调试信息
            return

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print(f"Error: Unable to load pixmap from {image_path}")  # 调试信息
            return

        label = QLabel()
        scaled_pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
        label.setFixedSize(100, 100)
        label.setFrameShape(QFrame.Box)
        label.mousePressEvent = lambda event, path=image_path: self.show_full_output(path)
        self.preview_layout.addWidget(label)

    def show_full_output(self, image_path):
        if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
            print(f"Error: {image_path} is missing or empty.")  # 调试信息
            placeholder_pixmap = QPixmap(600, 600)
            placeholder_pixmap.fill(Qt.lightGray)  # 占位符灰色背景
            self.output_image_label.setPixmap(placeholder_pixmap)
            return

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print(f"Error: Unable to load pixmap from {image_path}")  # 调试信息
            return

        scaled_pixmap = pixmap.scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.output_image_label.setPixmap(scaled_pixmap)

    def open_image(self, image_path):
        if os.path.exists(image_path):
            subprocess.Popen(['xdg-open', image_path])  # Linux示例，可替换为Windows或Mac指令

    def update_progress(self, message, value):
        self.progress_label.setText(message)
        self.progress_bar.setValue(value)

    def display_images(self, input_path, final_path):
        self.input_image_label.setPixmap(QPixmap(input_path).scaled(300, 300, Qt.KeepAspectRatio))
        self.output_image_label.setPixmap(QPixmap(final_path).scaled(300, 300, Qt.KeepAspectRatio))

    def store_results(self, input_path, final_path):
        self.result_images.append(final_path)

        # 更新滚动区域
        # row = len(self.result_images) // 3
        # col = len(self.result_images) % 3
        pixmap = QPixmap(final_path).scaled(200, 200, Qt.KeepAspectRatio)
        label = QLabel()
        label.setPixmap(pixmap)
        label.setFrameShape(QFrame.Box)
        # self.scroll_layout.addWidget(label, row, col)

    def save_results(self):
        if not self.result_images:
            self.progress_label.setText("No results to save.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Directory to Save Results")
        if folder:
            for idx, img_path in enumerate(self.result_images):
                img = QPixmap(img_path)
                save_path = os.path.join(folder, f"result_{idx + 1}.png")
                img.save(save_path)
            self.progress_label.setText("Results saved successfully.")

    def clear_results(self):
        self.result_images = []
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        self.file_label.setText("Input Image: Not selected")
        self.input_image_label.clear()

    def export_settings(self):
        settings = {
            "function": self.function_combo.currentText(),
            "positive_prompt": self.positive_prompt_input.text(),
            "negative_prompt": self.negative_prompt_input.text(),
            "scale": self.scale_slider.value(),
            "sampler": self.sampler_combo.currentText(),
            "steps": self.steps_input.value(),
            "strength": self.strength_input.value(),
            "noise": self.noise_input.value(),
            "noise_schedule": self.noise_schedule_combo.currentText(),
            "api_key": self.api_key_input.text(),
            "general_thresh": self.general_thresh_input.value(),
            "character_thresh": self.character_thresh_input.value(),
            "use_wd": self.use_wd_checkbox.isChecked(),
            "run_count": self.run_count_input.value(),
        }
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Settings", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(settings, f)
            self.progress_label.setText("Settings exported successfully.")

    def import_settings(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Settings", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'r') as f:
                settings = json.load(f)

            self.function_combo.setCurrentText(settings["function"])
            self.positive_prompt_input.setText(settings["positive_prompt"])
            self.negative_prompt_input.setText(settings["negative_prompt"])
            self.scale_slider.setValue(settings["scale"])
            self.sampler_combo.setCurrentText(settings["sampler"])
            self.steps_input.setValue(settings["steps"])
            self.strength_input.setValue(settings["strength"])
            self.noise_input.setValue(settings["noise"])
            self.noise_schedule_combo.setCurrentText(settings["noise_schedule"])
            self.api_key_input.setText(settings["api_key"])
            self.general_thresh_input.setValue(settings["general_thresh"])
            self.character_thresh_input.setValue(settings["character_thresh"])
            self.use_wd_checkbox.setChecked(settings["use_wd"])
            self.run_count_input.setValue(settings["run_count"])

            self.progress_label.setText("Settings imported successfully.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
