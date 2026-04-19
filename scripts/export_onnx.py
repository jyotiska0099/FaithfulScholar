# scripts/export_onnx.py
# Run this once to export the sentence transformer to ONNX and quantise it.
# Produces two files:
#   models/onnx/model.onnx          — full precision FP32
#   models/onnx/model_int8.onnx     — INT8 quantised

import os
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType

# Directories
ONNX_DIR = "models/onnx"
os.makedirs(ONNX_DIR, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_PATH = os.path.join(ONNX_DIR, "model.onnx")
INT8_PATH = os.path.join(ONNX_DIR, "model_int8.onnx")

# ── Export to ONNX ─────────────────────────────────────────────────────────────
print("Exporting model to ONNX...")
# ORTModelForFeatureExtraction handles the export automatically.
# export=True triggers the ONNX conversion from the HuggingFace model.
model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Save the exported model and tokenizer
model.save_pretrained(ONNX_DIR)
tokenizer.save_pretrained(ONNX_DIR)
print(f"ONNX model saved to {ONNX_DIR}")

# ── Quantise to INT8 ───────────────────────────────────────────────────────────
print("Quantising to INT8...")

# Find the exported .onnx file — optimum saves it as model.onnx
exported_path = os.path.join(ONNX_DIR, "model.onnx")

quantize_dynamic(
    model_input=exported_path,
    model_output=INT8_PATH,
    # QuantType.QUInt8 is recommended for CPU inference on most hardware
    weight_type=QuantType.QUInt8
)

print(f"INT8 model saved to {INT8_PATH}")
print("\nExport complete. Files produced:")
print(f"  FP32: {exported_path}")
print(f"  INT8: {INT8_PATH}")