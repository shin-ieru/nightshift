# sam3_helper.py

from transformers import Sam3Model, Sam3Processor
from PIL import Image
import torch
import numpy as np

# Use GPU if available, otherwise CPU
_device = "cuda" if torch.cuda.is_available() else "cpu"

_model: Sam3Model | None = None
_processor: Sam3Processor | None = None

MODEL_ID = "facebook/sam3"   # same checkpoint you used before


def load_sam3():
    """
    Lazy-load a single global SAM3 model + processor.
    """
    global _model, _processor

    if _model is None or _processor is None:
        # Processor handles image + text tokenization and post-processing
        _processor = Sam3Processor.from_pretrained(MODEL_ID)

        # Plain single-device load (no device_map="auto" so it works on CPU laptops)
        _model = Sam3Model.from_pretrained(MODEL_ID)
        _model.to(_device)

    return _model, _processor


def run_sam3_on_path(image_path: str, text_prompt: str):
    """
    Run SAM3 on a single image file with a single natural-language prompt.

    Returns a list of:
        {
          "label": <prompt>,
          "score": float,
          "bbox": [x, y, w, h],
          "mask": np.ndarray  # (H, W) bool/float mask
        }
    """
    model, processor = load_sam3()

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Prepare inputs and move them to the same device as the model
    inputs = processor(
        images=image,
        text=text_prompt,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process into instance segments (boxes + masks + scores)
    processed = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]

    detections = []
    boxes = processed.get("boxes", torch.empty(0))
    scores = processed.get("scores", torch.empty(0))
    masks = processed.get("masks", torch.empty(0))

    for box, score, mask in zip(boxes, scores, masks):
        x1, y1, x2, y2 = box.tolist()
        w = x2 - x1
        h = y2 - y1

        detections.append(
            {
                "label": text_prompt,
                "score": float(score),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "mask": mask.cpu().numpy(),  # H x W
            }
        )

    return detections