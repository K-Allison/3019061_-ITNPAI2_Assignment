"""
Flower Classifier Demo Application
=================================

A Python application that lets a user browse for an image, validates that it
is a supported image file, runs the same preprocessing used during testing
to ensure consistency, and then shows predictions from two saved models:

1. A Random Forest and SVM models saved with joblib
2. A MobileNetV2 model saved as a PyTorch checkpoint
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageTk, UnidentifiedImageError
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog
from tkinter import Tk, Frame, Label, Button, Text, filedialog, messagebox, StringVar
from tkinter import ttk
from torchvision import models, transforms


# ============================================================================
# Configuration
# ============================================================================

# saved model files live elsewhere.
RANDOMFOREST_MODEL_PATH = Path("models/flower_random_forest.pkl")
SVM_MODEL_PATH = Path("models/flower_svm.pkl")
MOBILENET_CHECKPOINT_PATH = Path("models/flower_mobilenetv2.pth")

# Image-processing parameters. These must match training/test-time preprocessing.
IMG_SIZE = (224, 224)
HIST_BINS = 16
SUPPORTED_EXTS = {".jpg"}

# ImageNet normalization used for MobileNetV2 inference.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Preview size inside the GUI.
PREVIEW_MAX_SIZE = (380, 380)


# ============================================================================
# Feature extraction for the classical baseline
# ============================================================================


def pil_to_rgb_array(pil_image: Image.Image, size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Convert a PIL image to a resized RGB numpy array.

    This mirrors the classical test-time preprocessing used in the colab notebook.
    """
    return np.array(pil_image.convert("RGB").resize(size), dtype=np.uint8)



def colour_hist_features(img_rgb: np.ndarray, bins: int = HIST_BINS) -> np.ndarray:
    """Create HSV colour histogram features.

    The image is converted from RGB to HSV, then a normalized histogram is built
    for each channel. The three histograms are concatenated into one vector.
    """
    hsv = rgb2hsv(img_rgb)
    features: List[float] = []

    for channel in range(3):
        hist, _ = np.histogram(
            hsv[:, :, channel],
            bins=bins,
            range=(0, 1),
            density=True,
        )
        features.extend(hist.tolist())

    return np.asarray(features, dtype=np.float32)



def hog_features(img_rgb: np.ndarray) -> np.ndarray:
    """Create HOG edge/gradient features.

    This matches the HOG settings used in the notebook baseline.
    """
    gray = rgb2gray(img_rgb)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    return np.asarray(features, dtype=np.float32)



def extract_classical_features(pil_image: Image.Image) -> np.ndarray:
    """Extract the combined colour histogram + HOG feature vector.

    Returns a 2D array with shape (1, n_features), because scikit-learn models
    expect a batch dimension even for a single sample.
    """
    img_rgb = pil_to_rgb_array(pil_image, size=IMG_SIZE)
    hist = colour_hist_features(img_rgb, bins=HIST_BINS)
    hog_vec = hog_features(img_rgb)
    features = np.concatenate([hist, hog_vec]).astype(np.float32)
    return features.reshape(1, -1)


# ============================================================================
# Model loading helpers
# ============================================================================


def extract_sklearn_classes(model) -> List[str]:
    """Return the class list from a saved scikit-learn model or pipeline.

    Works for plain estimators such as RandomForestClassifier and for Pipeline
    objects such as StandardScaler + SVC.
    """
    if hasattr(model, "classes_"):
        return list(model.classes_)

    if hasattr(model, "named_steps"):
        # Try the final step first, which is where classes_ usually lives.
        named_steps = model.named_steps
        if named_steps:
            last_step = list(named_steps.values())[-1]
            if hasattr(last_step, "classes_"):
                return list(last_step.classes_)

    raise ValueError(
        "Could not determine class names from the baseline model. "
        "Save a fitted estimator or fitted Pipeline."
    )



def load_baseline_model(model_path: Path):
    """Load the classical baseline model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Baseline model file not found: {model_path.resolve()}"
        )
    model = joblib.load(model_path)
    classes = extract_sklearn_classes(model)
    return model, classes



def load_mobilenet_model(checkpoint_path: Path):
    """Load the MobileNetV2 checkpoint and rebuild the model architecture."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"MobileNetV2 checkpoint file not found: {checkpoint_path.resolve()}"
        )

    # Always load to CPU so the app works even on machines without a GPU.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "num_classes" not in checkpoint or "class_names" not in checkpoint:
        raise KeyError(
            "Checkpoint is missing 'num_classes' or 'class_names'. "
            "Save the checkpoint using the structure discussed in Colab."
        )

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing 'model_state_dict'.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return model, class_names, transform


# ============================================================================
# Prediction helpers
# ============================================================================


def format_topk(classes: List[str], probs: np.ndarray, k: int = 3) -> str:
    """Return a readable top-k prediction string for the GUI."""
    k = min(k, len(classes), len(probs))
    top_indices = np.argsort(probs)[::-1][:k]

    lines = []
    for rank, idx in enumerate(top_indices, start=1):
        lines.append(f"{rank}. {classes[idx]}: {probs[idx]:.4f}")
    return "\n".join(lines)



def predict_with_baseline(model, classes: List[str], pil_image: Image.Image) -> Dict[str, object]:
    """Run inference using the saved classical model.

    The function supports both models with predict_proba (preferred) and models
    with only predict.
    """
    features = extract_classical_features(pil_image)

    # Top-1 prediction is always available.
    predicted_label = model.predict(features)[0]

    result: Dict[str, object] = {
        "top1_label": str(predicted_label),
        "top1_confidence": None,
        "top3_text": f"1. {predicted_label}",
    }

    # Probability output is available for Random Forest and for SVM when
    # probability=True was used during training.
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        # Use the model's own class order so probabilities align correctly.
        model_classes = extract_sklearn_classes(model)
        top_index = int(np.argmax(probs))
        result["top1_label"] = str(model_classes[top_index])
        result["top1_confidence"] = float(probs[top_index])
        result["top3_text"] = format_topk(model_classes, probs, k=3)

    return result



def predict_with_mobilenet(
    model: nn.Module,
    classes: List[str],
    transform,
    pil_image: Image.Image,
) -> Dict[str, object]:
    """Run inference using the trained MobileNetV2 model."""
    image_tensor = transform(pil_image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_index = int(np.argmax(probs))

    return {
        "top1_label": str(classes[top_index]),
        "top1_confidence": float(probs[top_index]),
        "top3_text": format_topk(classes, probs, k=3),
    }


# ============================================================================
# Image validation helpers
# ============================================================================


def validate_image_path(file_path: Path) -> None:
    """Raise a clear exception to user if the selected file is not a valid image."""
    if not file_path:
        raise ValueError("No file was selected.")

    if not file_path.exists():
        raise FileNotFoundError(f"Selected file does not exist: {file_path}")

    if file_path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(
            "Unsupported file type. Please choose a JPG, JPEG, PNG, BMP, WEBP, TIF, or TIFF image."
        )

    # Verify the file really is a readable image. PIL's verify() is strict and
    # catches many corrupted/invalid files.
    try:
        with Image.open(file_path) as test_image:
            test_image.verify()
    except UnidentifiedImageError as exc:
        raise ValueError("The selected file is not a valid image.") from exc
    except Exception as exc:
        raise ValueError(f"Unable to open the selected image: {exc}") from exc



def load_image_for_prediction(file_path: Path) -> Image.Image:
    """Open the image in RGB mode after validation succeeds."""
    validate_image_path(file_path)
    return Image.open(file_path).convert("RGB")


# ============================================================================
# Tkinter GUI application
# ============================================================================


class FlowerClassifierApp:
    """Desktop GUI application for comparing three flower classifiers."""

    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Flower Classifier Comparison")
        self.root.geometry("1200x720")
        self.root.minsize(1020, 640)

        # Hold a reference to the displayed PhotoImage so Tkinter does not
        # garbage-collect it and make the preview disappear.
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.current_image_path: Optional[Path] = None

        # Status text shown at the bottom of the window.
        self.status_var = StringVar(value="Loading models...")

        # Load models up front so the user sees any file/configuration problem
        # immediately rather than after choosing an image.
        self.baseline_model = None
        self.baseline_classes: List[str] = []
        self.mobilenet_model = None
        self.mobilenet_classes: List[str] = []
        self.mobilenet_transform = None

        self._build_layout()
        self._load_models()

    def _build_layout(self) -> None:
        """Create and place all GUI widgets."""
        # Main title
        title = Label(
            self.root,
            text="Flower Type Classifier",
            font=("Segoe UI", 20, "bold"),
            pady=10,
        )
        title.pack()

        subtitle = Label(
            self.root,
            text="Compare Random Forest, SVM, and MobileNetV2 on an uploaded flower image.",
            font=("Segoe UI", 10),
        )
        subtitle.pack()

        # Top control bar
        controls = Frame(self.root, padx=12, pady=12)
        controls.pack(fill="x")

        browse_btn = Button(
            controls,
            text="Browse for Image...",
            command=self.choose_image,
            width=18,
            font=("Segoe UI", 10, "bold"),
        )
        browse_btn.pack(side="left", padx=(0, 8))

        rerun_btn = Button(
            controls,
            text="Re-run Prediction",
            command=self.rerun_current_image,
            width=18,
            font=("Segoe UI", 10),
        )
        rerun_btn.pack(side="left")

        # Main content area
        content = Frame(self.root, padx=12, pady=8)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=2)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        # -------------------------
        # Left side: image preview
        # -------------------------
        left_frame = ttk.LabelFrame(content, text="Selected Image", padding=12)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=4)

        self.image_label = Label(left_frame, text="No image selected.", anchor="center")
        self.image_label.pack(fill="both", expand=True)

        self.file_info_label = Label(
            left_frame,
            text="",
            justify="left",
            anchor="w",
            font=("Segoe UI", 10),
            pady=8,
        )
        self.file_info_label.pack(fill="x")

        # -------------------------
        # Right side: predictions
        # -------------------------
        right_frame = ttk.LabelFrame(content, text="Predictions", padding=12)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=4)

        right_frame.grid_propagate(False)
        right_frame.configure(width=520)

        right_frame.columnconfigure(0, weight=1)
        right_frame.columnconfigure(1, weight=1)
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)

        # Random Forest box
        self.rf_text = Text(right_frame, height=14, wrap="word", font=("Consolas", 10))
        self.rf_text.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=(0, 10))

        # SVM box
        self.svm_text = Text(right_frame, height=14, wrap="word", font=("Consolas", 10))
        self.svm_text.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=(0, 10))

        # MobileNetV2 box spans both columns
        self.mobilenet_text = Text(right_frame, height=14, wrap="word", font=("Consolas", 10))
        self.mobilenet_text.grid(row=1, column=0, columnspan=2, sticky="nsew")

        # Default text
        self._write_text(self.rf_text, "Random Forest results will appear here.")
        self._write_text(self.svm_text, "SVM results will appear here.")
        self._write_text(self.mobilenet_text, "MobileNetV2 results will appear here.")

        # Bottom status bar
        status_bar = Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            relief="sunken",
            padx=8,
            pady=6,
        )
        status_bar.pack(fill="x", side="bottom")

    def _load_models(self) -> None:
        """Load model files and report any configuration errors clearly."""
        try:
            self.rf_model, self.rf_classes = load_baseline_model(RANDOMFOREST_MODEL_PATH)
            self.svm_model, self.svm_classes = load_baseline_model(SVM_MODEL_PATH)
            self.mobilenet_model, self.mobilenet_classes, self.mobilenet_transform = load_mobilenet_model(
                MOBILENET_CHECKPOINT_PATH
            )
            self.status_var.set("Models loaded successfully. Choose an image to begin.")
        except Exception as exc:
            self.status_var.set("Failed to load models.")
            messagebox.showerror(
                "Model Loading Error",
                f"The application could not load the saved models.\n\n{exc}",
            )
            # Re-raise so the issue is visible during development.
            raise

    @staticmethod
    def _write_text(widget: Text, content: str) -> None:
        """Safely replace the contents of a Tkinter Text widget."""
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.config(state="disabled")

    def choose_image(self) -> None:
        """Open a file dialog and process the selected image."""
        file_path = filedialog.askopenfilename(
            title="Choose a flower image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )

        # The user may cancel the dialog; that is not an error.
        if not file_path:
            self.status_var.set("No file selected.")
            return

        self.process_image(Path(file_path))

    def rerun_current_image(self) -> None:
        """Run both models again on the currently loaded image."""
        if self.current_image_path is None:
            messagebox.showinfo("No Image", "Please choose an image first.")
            return
        self.process_image(self.current_image_path)

    def process_image(self, file_path: Path) -> None:
        """Validate the file, display it, and show predictions from all models."""
        try:
            self.status_var.set("Validating image...")
            pil_image = load_image_for_prediction(file_path)

            self.current_image_path = file_path
            self._display_image_preview(pil_image)
            self._update_file_info(file_path, pil_image)

            self.status_var.set("Running Random Forest model...")
            rf_result = predict_with_baseline(
                self.rf_model,
                self.rf_classes,
                pil_image,
            )

            self.status_var.set("Running SVM model...")
            svm_result = predict_with_baseline(
                self.svm_model,
                self.svm_classes,
                pil_image,
            )

            self.status_var.set("Running MobileNetV2...")
            mobilenet_result = predict_with_mobilenet(
                self.mobilenet_model,
                self.mobilenet_classes,
                self.mobilenet_transform,
                pil_image,
            )

            rf_output = self._format_result_block(
                title="Random Forest Baseline",
                model_path=RANDOMFOREST_MODEL_PATH,
                result=rf_result,
            )

            svm_output = self._format_result_block(
                title="SVM Baseline",
                model_path=SVM_MODEL_PATH,
                result=svm_result,
            )

            mobilenet_output = self._format_result_block(
                title="MobileNetV2",
                model_path=MOBILENET_CHECKPOINT_PATH,
                result=mobilenet_result,
            )

            # Write each model into its own text box
            self._write_text(self.rf_text, rf_output)
            self._write_text(self.svm_text, svm_output)
            self._write_text(self.mobilenet_text, mobilenet_output)

            self.status_var.set(f"Prediction complete for: {file_path.name}")

        except Exception as exc:
            self.status_var.set("Prediction failed.")
            self._write_text(self.rf_text, "Random Forest results will appear here.")
            self._write_text(self.svm_text, "SVM results will appear here.")
            self._write_text(self.mobilenet_text, "MobileNetV2 results will appear here.")

            messagebox.showerror(
                "Prediction Error",
                f"The application could not process the selected file.\n\n{exc}",
            )

            print("\n--- Full error traceback ---")
            print(traceback.format_exc())

    def _display_image_preview(self, pil_image: Image.Image) -> None:
        """Resize and display the chosen image inside the GUI."""
        preview_image = pil_image.copy()
        preview_image.thumbnail(PREVIEW_MAX_SIZE)
        self.current_photo = ImageTk.PhotoImage(preview_image)
        self.image_label.config(image=self.current_photo, text="")

    def _update_file_info(self, file_path: Path, pil_image: Image.Image) -> None:
        """Show basic metadata about the selected file."""
        info = (
            f"File: {file_path.name}\n"
            f"Original size: {pil_image.width} x {pil_image.height} pixels\n"
            f"Processed size: {IMG_SIZE[0]} x {IMG_SIZE[1]} pixels"
        )
        self.file_info_label.config(text=info)

    @staticmethod
    def _format_result_block(title: str, model_path: Path, result: Dict[str, object]) -> str:
        """Turn a model result dictionary into readable text for the GUI."""
        confidence = result.get("top1_confidence")
        if confidence is None:
            confidence_text = "Not available (model does not expose probabilities)"
        else:
            confidence_text = f"{confidence:.4f}"

        return (
            f"{title}\n"
            f"{'=' * len(title)}\n"
            f"Model file: {model_path.name}\n\n"
            f"Top-1 prediction: {result['top1_label']}\n"
            f"Top-1 confidence: {confidence_text}\n\n"
            f"Top predictions:\n{result['top3_text']}\n"
        )


# ============================================================================
# Main entry point
# ============================================================================


def main() -> None:
    """Start the desktop application."""
    root = Tk()
    app = FlowerClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
