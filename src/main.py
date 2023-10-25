import os
import argparse
from environ import logging, get_root_dir
from ocr_parser.processor import extract_images

log = logging.getLogger(__name__)
current_module = __import__(__name__)


def try_tesseract_parser(imgfile):
    from ocr_parser.base import TesseractLayoutParser
    parser = TesseractLayoutParser()
    for img in extract_images(imgfile):
        res = parser.parse(img)
        log.info(res)
        for f in (
                TesseractLayoutParser.TesseractFeatureType.BLOCK, TesseractLayoutParser.TesseractFeatureType.PAGE,
                TesseractLayoutParser.TesseractFeatureType.PARA, TesseractLayoutParser.TesseractFeatureType.LINE,
                TesseractLayoutParser.TesseractFeatureType.WORD):
            log.info('%s: %s' % (f, parser.gather_data(res, f)))


def try_detectron_parser(imgfile):
    from ocr_parser.base import Detectron2LayoutParser
    parser = Detectron2LayoutParser()
    for img in extract_images(imgfile):
        res = parser.parse(img)
        log.info(res)
        # log.info(parser.gather_data(res))


def try_paddle_parser(imgfile):
    from ocr_parser.base import PaddleOCRParser
    parser = PaddleOCRParser()
    for img in extract_images(imgfile):
        res = parser.parse(img)
        log.info(res)


def try_gcv_parser(imgfile):
    from ocr_parser.base import GCVLayoutParser
    parser = GCVLayoutParser()
    for img in extract_images(imgfile):
        res = parser.parse(img)
        log.info(res.full_text_annotation.text)
        for f in (
                GCVLayoutParser.GCVFeatureType.BLOCK, GCVLayoutParser.GCVFeatureType.PAGE,
                GCVLayoutParser.GCVFeatureType.PARA, GCVLayoutParser.GCVFeatureType.SYMBOL,
                GCVLayoutParser.GCVFeatureType.WORD):
            log.info('%s: %s' % (f, parser.gather_data(res, f).get_texts()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='OCR POC')
    parser.add_argument('-f', '--img_file', dest='img_file', required=False,
                        default=os.path.join(get_root_dir(), 'images', 'acordjpeg.jpg'))
    parser.add_argument('-p', '--parser', dest='parser', required=False,
                        default='GCV')
    args = parser.parse_args()

    try:
        try_method = getattr(current_module, f'try_{args.parser.lower()}_parser')
        try_method(args.img_file)
    except:
        log.exception(f'Error trying {args.parser}')
