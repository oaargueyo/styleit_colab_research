from fastapi import FastAPI, File, UploadFile, Query
import shutil
import os
import styleit_cv

import torch, detectron2

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

app = FastAPI()

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "PLEASE_SET_SERPER_API_KEY")  # Needs an enterprise api key for Serper
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID", 'PLEASE_SET_IMGUR_CLIENT_ID') # Needs a client id for imgur 


# Create a directory to save uploaded images
UPLOAD_DIR = "/home/appuser/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

image_uploader = styleit_cv.ImgurUploader(client_id=IMGUR_CLIENT_ID)
image_detector = styleit_cv.Detectron2Processor()
image_search = styleit_cv.ReverseImageSearch(api_key=SERPER_API_KEY)

@app.post("/image-search/", tags=["Style IT Reverse Image Search API"])
async def process_uploaded_image(file: UploadFile = File(...),
                                 max_results: int = Query(100, description="Maximum number of results to return")):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save the uploaded file to the server
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

        # Process the image with the detectron logic
        image = image_uploader.upload_image(image_path=file_path, title='Test Image')
        outputs, image_cv = image_detector.run_inference(image_path=file_path)
        cropper = styleit_cv.ImageCropper(image=image_cv, client_id=IMGUR_CLIENT_ID)
        sections = cropper.crop_instances(outputs=outputs, cfg=image_detector.cfg)
        human_sections = [item for item in sections if "person" in item[2]]
        human_sections_links = []
        for hs in human_sections:
            human_sections_links += cropper.split_human_image(image_path=hs[1])
            human_sections_links.append(hs[0])

        search_results = image_search.reverse_image_search(image_urls=human_sections_links)

    return {"search_results": search_results[:max_results]}

