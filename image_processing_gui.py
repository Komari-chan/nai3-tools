import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QSlider, QPushButton, QComboBox,
    QCheckBox, QFileDialog, QHBoxLayout, QSpinBox, QDoubleSpinBox, QProgressBar, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
import importlib.util


class ProcessThread(QThread):
    progress_signal = pyqtSignal(str, int)  # (Progress message, Progress percentage)
    image_paths_signal = pyqtSignal(str, str)  # (Input image path, Final image path)

    def __init__(self, module_name, params):
        super().__init__()
        self.module_name = module_name
        self.params = params

    def run(self):
        # 动态加载模块
        spec = importlib.util.spec_from_file_location(self.module_name, f"./{self.module_name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 使用后端函数
        try:
            input_image = self.params['image_path']
            module.process_image_with_novelai(**self.params)
            # 假设输出文件路径固定
            final_image_path = "output/final_image_with_novelai_and_auto_prompts.png"
            self.image_paths_signal.emit(input_image, final_image_path)
        except Exception as e:
            self.progress_signal.emit(f"Error: {str(e)}", 100)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("NovelAI Image Processor")
        self.setGeometry(100, 100, 700, 900)
        layout = QVBoxLayout()

        # Input file section
        self.file_label = QLabel("Input Image: Not selected")
        layout.addWidget(self.file_label)

        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.select_file)
        layout.addWidget(self.file_button)

        # Functionality selection
        self.function_label = QLabel("Select Functionality:")
        layout.addWidget(self.function_label)

        self.function_combo = QComboBox()
        self.function_combo.addItems(["1-2.5倍", "5-5倍", "5-5倍BZ"])
        layout.addWidget(self.function_combo)

        # Positive prompt
        self.positive_prompt_label = QLabel("Positive Prompt:")
        layout.addWidget(self.positive_prompt_label)
        self.positive_prompt_input = QLineEdit()
        layout.addWidget(self.positive_prompt_input)

        # Negative prompt
        self.negative_prompt_label = QLabel("Negative Prompt (default provided):")
        layout.addWidget(self.negative_prompt_label)
        self.negative_prompt_input = QLineEdit(
            "nsfw, lowres, {bad}, error, fewer, extra, missing, "
            "worst quality, jpeg artifacts, bad quality, watermark, "
            "unfinished, displeasing, chromatic aberration, signature, "
            "extra digits, artistic error, username, scan, [abstract]"
        )
        layout.addWidget(self.negative_prompt_input)

        # Scale
        self.scale_label = QLabel("Scale (0-10):")
        layout.addWidget(self.scale_label)
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(0, 10)
        self.scale_slider.setValue(5)
        layout.addWidget(self.scale_slider)

        # Sampler
        self.sampler_label = QLabel("Sampler:")
        layout.addWidget(self.sampler_label)
        self.sampler_combo = QComboBox()
        self.sampler_combo.addItems(["k_dpmpp_2s_ancestral", "k_dpmpp_2m_sde"])  # 可以扩展选项
        layout.addWidget(self.sampler_combo)

        # Steps
        self.steps_label = QLabel("Steps (1-28):")
        layout.addWidget(self.steps_label)
        self.steps_input = QSpinBox()
        self.steps_input.setRange(1, 28)
        self.steps_input.setValue(28)
        layout.addWidget(self.steps_input)

        # Strength
        self.strength_label = QLabel("Strength (0.0-0.99):")
        layout.addWidget(self.strength_label)
        self.strength_input = QDoubleSpinBox()
        self.strength_input.setRange(0.0, 0.99)
        self.strength_input.setValue(0.3)
        self.strength_input.setSingleStep(0.01)
        layout.addWidget(self.strength_input)

        # Noise
        self.noise_label = QLabel("Noise (0.0-0.99):")
        layout.addWidget(self.noise_label)
        self.noise_input = QDoubleSpinBox()
        self.noise_input.setRange(0.0, 0.99)
        self.noise_input.setValue(0.0)
        self.noise_input.setSingleStep(0.01)
        layout.addWidget(self.noise_input)

        # Noise schedule
        self.noise_schedule_label = QLabel("Noise Schedule:")
        layout.addWidget(self.noise_schedule_label)
        self.noise_schedule_combo = QComboBox()
        self.noise_schedule_combo.addItems(["karras", "native"])  # 可以扩展选项
        layout.addWidget(self.noise_schedule_combo)

        # API Key
        self.api_key_label = QLabel("API Key:")
        layout.addWidget(self.api_key_label)
        self.api_key_input = QLineEdit()
        layout.addWidget(self.api_key_input)

        # Use WD
        self.use_wd_checkbox = QCheckBox("Use WD Tagger")
        self.use_wd_checkbox.stateChanged.connect(self.toggle_wd_inputs)
        layout.addWidget(self.use_wd_checkbox)

        # WD Thresholds
        self.general_thresh_label = QLabel("General Threshold (0.0-1.0):")
        layout.addWidget(self.general_thresh_label)
        self.general_thresh_input = QDoubleSpinBox()
        self.general_thresh_input.setRange(0.0, 1.0)
        self.general_thresh_input.setValue(0.35)
        self.general_thresh_input.setSingleStep(0.01)
        self.general_thresh_input.setEnabled(False)
        layout.addWidget(self.general_thresh_input)

        self.character_thresh_label = QLabel("Character Threshold (0.0-1.0):")
        layout.addWidget(self.character_thresh_label)
        self.character_thresh_input = QDoubleSpinBox()
        self.character_thresh_input.setRange(0.0, 1.0)
        self.character_thresh_input.setValue(0.9)
        self.character_thresh_input.setSingleStep(0.01)
        self.character_thresh_input.setEnabled(False)
        layout.addWidget(self.character_thresh_input)

        # Progress bar
        self.progress_label = QLabel("Progress:")
        layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Submit Button
        self.submit_button = QPushButton("Process Image")
        self.submit_button.clicked.connect(self.process_image)
        layout.addWidget(self.submit_button)

        # Image Display
        self.input_image_display = QLabel("Input Image")
        self.input_image_display.setFrameShape(QFrame.Box)
        layout.addWidget(self.input_image_display)

        self.output_image_display = QLabel("Final Image")
        self.output_image_display.setFrameShape(QFrame.Box)
        layout.addWidget(self.output_image_display)

        self.setLayout(layout)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.file_label.setText(f"Input Image: {file_path}")
            self.input_image_path = file_path
            self.input_image_display.setPixmap(QPixmap(file_path).scaled(300, 300, Qt.KeepAspectRatio))

    def toggle_wd_inputs(self, state):
        enabled = state == Qt.Checked
        self.general_thresh_input.setEnabled(enabled)
        self.character_thresh_input.setEnabled(enabled)

    def process_image(self):
        module_name = {
            "1-2.5倍": "img2imgWD",
            "5-5倍": "img2imgWD_5",
            "5-5倍BZ": "img2imgWD_BZ",
        }[self.function_combo.currentText()]

        params = {
            "image_path": self.file_label.text().split(": ")[1],
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
        }

        self.thread = ProcessThread(module_name, params)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.image_paths_signal.connect(self.display_images)
        self.thread.start()

    def update_progress(self, message, value):
        self.progress_label.setText(message)
        self.progress_bar.setValue(value)

    def display_images(self, input_path, final_path):
        self.input_image_display.setPixmap(QPixmap(input_path).scaled(300, 300, Qt.KeepAspectRatio))
        self.output_image_display.setPixmap(QPixmap(final_path).scaled(300, 300, Qt.KeepAspectRatio))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
