import os
import cv2
import json
import pyimgur
import torch
import requests
import pprint
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
)
from detectron2.structures import Boxes
from serpapi import GoogleSearch
UPLOAD_DIR = "/home/appuser/app/uploads"


ALLOWED_DOMAINS = [
    'saltandseakauai',
    'amazon',
    'ebay',
    'walmart',
    'target',
    'bestbuy',
    'homedepot',
    'macys',
    'costco',
    'lowes',
    'kohl',
]


class Detectron2Processor:
    def __init__(self):
        self.cfg = get_cfg()
        self._initialize_model()

    def _initialize_model(self):
        """Sets up the Detectron2 model and predictor."""
        self.cfg.MODEL.DEVICE = 'cpu'
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)

    def run_inference(self, image_path):
        """Runs inference on the image."""
        im = cv2.imread(image_path)
        outputs = self.predictor(im)
        return outputs, im

    def visualize_predictions(self, outputs, image):
        """Visualizes the predictions on the image."""
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return out.get_image()[:, :, ::-1]


class ImgurUploader:
    def __init__(self, client_id):
        self.client_id = client_id
        self.im_client = pyimgur.Imgur(self.client_id)

    def upload_image(self, image_path, title="Uploaded with PyImgur"):
        """Uploads an image to Imgur and returns the image URL."""
        uploaded_image = self.im_client.upload_image(f"{image_path}", title=title)
        return uploaded_image.link


class ImageCropper:
    def __init__(self, image, client_id):
        self.image = image
        self.uploader = ImgurUploader(client_id=client_id)

    def crop_instances(self, outputs, cfg):
        from detectron2.structures import Boxes

        # Extract the instances
        instances = outputs["instances"].to("cpu")
        classes = instances.pred_classes  # Class indices of detected objects
        #num_instances = len(instances)
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        class_names = metadata.get("thing_classes", [])  # List of class names

        """Crops the detected instances and returns the cropped image paths."""
        imgur_links = []
        boxes = instances.pred_boxes if instances.has("pred_boxes") else Boxes()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cropped_image = self.image[y1:y2, x1:x2]
            class_index = classes[i]
            class_label = class_names[class_index] if class_index < len(class_names) else "unknown"
            output_path = f"{UPLOAD_DIR}/cropped_instance_{class_label}_{i + 1}.jpg"
            cv2.imwrite(output_path, cropped_image)
            imgur_link = self.uploader.upload_image(output_path, title=f"Instance: {class_label}")
            imgur_links.append((imgur_link, output_path , class_label))
            print(f"Saved and uploaded: {output_path}, Link: {imgur_link}")
        return imgur_links

    def split_human_image(self, image_path):
        """Splits the human image into top, middle, and bottom parts and uploads to Imgur."""
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # Calculate the split points
        top_split = int(0.25 * height)
        bottom_split = int(0.75 * height)

        # Slice the image into three parts
        top_image = image[:top_split, :]
        middle_image = image[top_split:bottom_split, :]
        bottom_image = image[bottom_split:, :]

        # Extract the last part of the filename without the directory path and extension
        filename = os.path.splitext(os.path.basename(image_path))[0]

        # Save the images with the new names
        cv2.imwrite(f"{UPLOAD_DIR}/{filename}_top.jpg", top_image)
        cv2.imwrite(f"{UPLOAD_DIR}/{filename}_middle.jpg", middle_image)
        cv2.imwrite(f"{UPLOAD_DIR}/{filename}_bottom.jpg", bottom_image)

        imgur_links = [
            self.uploader.upload_image(f"{UPLOAD_DIR}/{filename}_top.jpg", title=f"{filename}_top"),
            self.uploader.upload_image(f"{UPLOAD_DIR}/{filename}_middle.jpg", title=f"{filename}_middle"),
            self.uploader.upload_image(f"{UPLOAD_DIR}/{filename}_bottom.jpg", title=f"{filename}_bottom")
        ]

        return imgur_links


class ReverseImageSearch:
    def __init__(self, api_key):
        self.api_key = api_key

    @staticmethod
    def filter_us_retailers_and_marketplaces(results):
        """Filters results to include only US-based retailers and marketplaces."""
        # LOGIC FOR RETAILERS WAS REMOVED
        return results

    def request_to_serper(self, image_url):
        """Uploads an image URL to Serper API."""
        url = "https://google.serper.dev/lens"
        payload = json.dumps({"url": image_url, "location": "United States"})
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, data=payload)
        return response.json()

    def reverse_image_search(self, image_urls):
        """Performs reverse image search on a list of image URLs."""
        search_results = []
        for image_url in image_urls:
            response = self.request_to_serper(image_url)
            filtered_response = self.filter_us_retailers_and_marketplaces(response.get("organic", []))
            search_results.append(filtered_response)
        return search_results
