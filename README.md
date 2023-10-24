**Layout Parser based OCR POC**

To create testing environment with conda:

_conda env create -f main_env.yml_

Parser choices defined ocr_parser.base:
  - TesseractLayoutParser
  - Detectron2LayoutParser
  - PaddleOCRParser
  - GCVLayoutParser

Note: Google Cloud Vision API key is required to use GCVLayoutParser. 
It should be stored in the path defined by environment variable GOOGLE_APPLICATION_CREDENTIALS and defaulted to credentials\gcv.json

Sample usage:

_python main.py -f images/acordjpeg.jpg -p tesseract_