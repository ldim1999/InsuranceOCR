from abc import ABCMeta, abstractmethod

import layoutparser as lp
from layoutparser.ocr.tesseract_agent import TesseractFeatureType
from layoutparser import GCVFeatureType

class LayoutParser(object, metaclass=ABCMeta):
    @abstractmethod
    def parse(self, image):
        pass

    @abstractmethod
    def gather_data(self, detect_res, agg_level):
        pass


class TesseractLayoutParser(LayoutParser):

    def __init__(self):
        self.ocr_agent = lp.TesseractAgent()

    def parse(self, image):
        detect_res = self.ocr_agent.detect(image, return_response=True)
        return detect_res

    def gather_data(self, detect_res, agg_level):
        return self.ocr_agent.gather_data(detect_res, agg_level)


class Detectron2LayoutParser(LayoutParser):

    def __init__(self):
        self.ocr_agent = lp.Detectron2LayoutModel(
            config_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',  # In model catalog
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},  # In model`label_map`
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]  # Optional
        )

    def parse(self, image):
        detect_res = self.ocr_agent.detect(image)
        return detect_res

    def gather_data(self, detect_res, agg_level):
        return self.ocr_agent.gather_output(detect_res)


class PaddleOCRParser(LayoutParser):

    def __init__(self):
        self.ocr_agent = lp.PaddleDetectionLayoutModel(
            config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config",
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            enforce_cpu=False)

    def parse(self, image):
        detect_res = self.ocr_agent.detect(image)
        return detect_res

    def gather_data(self, detect_res, agg_level):
        return self.ocr_agent.gather_output(detect_res)


class GCVLayoutParser(LayoutParser):

    def __init__(self):
        self.ocr_agent = lp.GCVAgent(languages=['en'])

    def parse(self, image):
        detect_res = self.ocr_agent.detect(image, return_response=True)
        return detect_res

    def gather_data(self, detect_res, agg_level):
        return self.ocr_agent.gather_full_text_annotation(detect_res, agg_level)
