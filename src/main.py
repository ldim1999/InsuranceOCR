import os
import cv2
from environ import logging, get_root_dir
from ocr_parser.base import lp, TesseractLayoutParser, TesseractFeatureType, \
    Detectron2LayoutParser, PaddleOCRParser, GCVLayoutParser, GCVFeatureType

log = logging.getLogger(__name__)


def test_tesseract_parser(img):
    parser = TesseractLayoutParser()
    res = parser.parse(img)
    log.info(res)
    for f in (
            TesseractFeatureType.BLOCK, TesseractFeatureType.PAGE,
            TesseractFeatureType.PARA, TesseractFeatureType.LINE,
            TesseractFeatureType.WORD):
        print('%s: %s' % (f,parser.gather_data(res, f)))

def test_detectron_parser(img):
    parser = Detectron2LayoutParser()
    res = parser.parse(img)
    log.info(res)
    #log.info(parser.gather_data(res))

def test_paddle_parser(img):
    parser = PaddleOCRParser()
    res = parser.parse(img)
    log.info(res)

def test_gcv_parser(img):
    parser = GCVLayoutParser()
    res = parser.parse(img)
    log.info(res.full_text_annotation.text)
    for f in (
            GCVFeatureType.BLOCK, GCVFeatureType.PAGE,
            GCVFeatureType.PARA, GCVFeatureType.SYMBOL,
            GCVFeatureType.WORD):
        print('%s: %s' % (f,parser.gather_data(res, f).get_texts()))

if __name__ == '__main__':
    imgPath = os.path.join(get_root_dir(), 'images', 'acordjpeg.jpg')
    img = cv2.imread(imgPath)
    test_gcv_parser(img)
    #test_detectron_parser(img)
