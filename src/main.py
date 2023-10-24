import os
import cv2
import argparse
from environ import logging, get_root_dir
from ocr_parser.base import TesseractLayoutParser, TesseractFeatureType, \
    Detectron2LayoutParser, PaddleOCRParser, GCVLayoutParser, GCVFeatureType

log = logging.getLogger(__name__)
current_module = __import__(__name__)


def try_tesseract_parser(img):
    parser = TesseractLayoutParser()
    res = parser.parse(img)
    log.info(res)
    for f in (
            TesseractFeatureType.BLOCK, TesseractFeatureType.PAGE,
            TesseractFeatureType.PARA, TesseractFeatureType.LINE,
            TesseractFeatureType.WORD):
        log.info('%s: %s' % (f, parser.gather_data(res, f)))


def try_detectron_parser(img):
    parser = Detectron2LayoutParser()
    res = parser.parse(img)
    log.info(res)
    # log.info(parser.gather_data(res))


def try_paddle_parser(img):
    parser = PaddleOCRParser()
    res = parser.parse(img)
    log.info(res)


def try_gcv_parser(img):
    parser = GCVLayoutParser()
    res = parser.parse(img)
    log.info(res.full_text_annotation.text)
    for f in (
            GCVFeatureType.BLOCK, GCVFeatureType.PAGE,
            GCVFeatureType.PARA, GCVFeatureType.SYMBOL,
            GCVFeatureType.WORD):
        log.info('%s: %s' % (f, parser.gather_data(res, f).get_texts()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='OCR POC')
    parser.add_argument('-f', '--img_file', dest='img_file', required=False,
                        default=os.path.join(get_root_dir(), 'images', 'acordjpeg.jpg'))
    parser.add_argument('-p', '--parser', dest='parser', required=False,
                        default='GCV')
    args = parser.parse_args()

    img = cv2.imread(args.img_file)

    try:
        try_method = getattr(current_module, f'try_{args.parser.lower()}_parser')
        try_method(img)
    except:
        log.exception(f'Error trying {args.parser}')
