import os
import urllib.request

import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import supervision as sv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
from dataclasses import dataclass

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import cv2

torch.use_deterministic_algorithms(False)

import supervision as sv
from segment_anything import SamPredictor

import numpy as np
from autodistill.detection import CaptionOntology, DetectionBaseModel

from autodistill_grounded_sam.helpers import (
    load_SAM
)

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class SegmentAnything(DetectionBaseModel):
    ontology: CaptionOntology
    sam_predictor: SamPredictor
    box_threshold: float
    text_threshold: float

    def __init__(self, ontology: CaptionOntology, box_threshold=0.35, text_threshold=0.25):
        self.ontology = ontology
        self.predictor = load_SAM()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def predict(self, input: str) -> sv.Detections:
        masks = self.predictor.generate(cv2.imread(input))

        results = sv.Detections.from_sam(masks)

        results.class_id = np.array([0] * len(results.mask[0]))

        return results

def load_SAM():
    # Check if segment-anything library is already installed

    AUTODISTILL_CACHE_DIR = os.path.expanduser("~/.cache/autodistill")
    SAM_CACHE_DIR = os.path.join(AUTODISTILL_CACHE_DIR, "segment_anything")
    SAM_CHECKPOINT_PATH = os.path.join(SAM_CACHE_DIR, "sam_vit_h_4b8939.pth")

    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

    # Create the destination directory if it doesn't exist
    os.makedirs(os.path.dirname(SAM_CHECKPOINT_PATH), exist_ok=True)

    # Download the file if it doesn't exist
    if not os.path.isfile(SAM_CHECKPOINT_PATH):
        urllib.request.urlretrieve(url, SAM_CHECKPOINT_PATH)

    SAM_ENCODER_VERSION = "vit_h"

    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
        device=DEVICE
    )
    sam_predictor = SamAutomaticMaskGenerator(sam)

    return sam_predictor