import os
import json
import gradio as gr
import torch
import re
from inference import create_inference_pipeline  

# --- Image Labeler with Inference Pipeline ---

class ImageLabeler:
    def __init__(self, model_path, labels_json_path, img_width, img_height, folder_path):
        self.current_index = 0
        self.image_paths = []
        self.labels = {}
        self.current_path = ""
        self.json_file = labels_json_path
        self.folder_path = folder_path

        # Model parameters
        self.img_width = img_width
        self.img_height = img_height
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create the inference pipeline
        self.predict_fn = create_inference_pipeline(model_path, img_width, img_height)
        
        # Load existing labels and characters
        self.load_existing_labels()
        self.load_images()

    def load_existing_labels(self):
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r') as f:
                    self.labels = json.load(f)
            except Exception as e:
                print(f"Error loading existing labels: {e}")
                self.labels = {}

    def find_next_unlabeled_index(self):
        for i in range(len(self.image_paths)):
            index = (self.current_index + i) % len(self.image_paths)
            if self.image_paths[index] not in self.labels:
                return index
        return -1

    def find_previous_unlabeled_index(self):
        for i in range(len(self.image_paths)):
            index = (self.current_index - i) % len(self.image_paths)
            if self.image_paths[index] not in self.labels:
                return index
        return -1

    def load_images(self):
        valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
        self.image_paths = [
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if f.lower().endswith(valid_extensions)
        ]
        self.current_index = self.find_next_unlabeled_index()
        self.current_path = "" if self.current_index == -1 else self.image_paths[self.current_index]

    def next_image(self, label):
        if not self.current_path:
            return "", "All images are labeled.", gr.update(value="")

        if not self.is_valid_label(label):
            return self.current_path, f"Invalid label: {label}. Must be 4 numeric characters.", gr.update(value=label)

        self.labels[self.current_path] = label
        with open(self.json_file, 'w') as f:
            json.dump(self.labels, f, indent=4)

        self.current_index = self.find_next_unlabeled_index()
        self.current_path = "" if self.current_index == -1 else self.image_paths[self.current_index]

        predicted_label = self.predict_fn(self.current_path) if self.predict_fn else ""
        return self.current_path, f"Image {self.current_index + 1} of {len(self.image_paths)} ({len(self.labels)} labeled)", gr.update(value=predicted_label)

    def previous_image(self):
        self.current_index = self.find_previous_unlabeled_index()
        self.current_path = "" if self.current_index == -1 else self.image_paths[self.current_index]
        predicted_label = self.predict_fn(self.current_path) if self.predict_fn else ""
        return self.current_path, f"Image {self.current_index + 1} of {len(self.image_paths)} ({len(self.labels)} labeled)", gr.update(value=predicted_label)

    def is_valid_label(self, label):
        return bool(re.fullmatch(r"[0-9]{4}", label))

    def update_ui(self):
        predicted_label = self.predict_fn(self.current_path) if self.current_path and self.predict_fn else ""
        return (
            self.current_path,
            f"Image {self.current_index + 1} of {len(self.image_paths)} ({len(self.labels)} labeled)" if self.current_path else f"All images are labeled ({len(self.labels)} labeled)",
            gr.update(interactive=True, value=predicted_label),
            gr.update(interactive=True),
            gr.update(interactive=True)
        )

def create_app():
    model_path = "weights/model_0.99.pth"
    labels_json_path = "labels.json"
    img_width = 200
    img_height = 50
    folder_path = 'dataset'
    labeler = ImageLabeler(model_path, labels_json_path, img_width, img_height, folder_path)

    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                counter = gr.Textbox(label="Progress", interactive=False)
        with gr.Row():
            image_display = gr.Image(type="filepath", label="Current Image")
        with gr.Row():
            label_input = gr.Textbox(label="Enter label (4 numeric characters)", interactive=False, autofocus=True)
        with gr.Row():
            prev_btn = gr.Button("Previous", interactive=False)
            next_btn = gr.Button("Next", interactive=False)
        app.load(fn=labeler.update_ui, inputs=None, outputs=[image_display, counter, label_input, prev_btn, next_btn])
        label_input.submit(fn=labeler.next_image, inputs=[label_input], outputs=[image_display, counter, label_input])
        next_btn.click(fn=labeler.next_image, inputs=[label_input], outputs=[image_display, counter, label_input])
        prev_btn.click(fn=labeler.previous_image, outputs=[image_display, counter, label_input])
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(share=False, debug=True)
