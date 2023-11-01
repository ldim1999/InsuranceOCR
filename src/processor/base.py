from google.cloud import documentai_v1
from abc import ABCMeta, abstractmethod
import io
from environ import logging

log = logging.getLogger(__name__)

GC_PROJECT_ID = '841544667388'  # 'weighty-nation-402822'
GCD_AI_PROCESSOR_ID = '4f14f8ae324646a1'


class BaseDocumentProcessor(object, metaclass=ABCMeta):
    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def process(self):
        pass


class GCDAI_Processor(BaseDocumentProcessor):
    def __init__(self):
        self.prepare()

    def prepare(self):
        self.client = documentai_v1.DocumentProcessorServiceClient()
        # self.processor = self.client.create_processor(request=request)

    def get_image_bytes(self, img):
        output = io.BytesIO()
        img.save(output, format=img.format)
        return output.getvalue()


    def process(self, img):
        raw_document = documentai_v1.RawDocument(mime_type="image/%s" % img.format.lower())
        raw_document.content = self.get_image_bytes(img) #base64.b64encode

        name = self.client.processor_path(GC_PROJECT_ID, "us", GCD_AI_PROCESSOR_ID)

        request = documentai_v1.ProcessRequest(
            name=name,
            raw_document=raw_document
        )

        log.info('Sending request to %s', name)

        response = self.client.process_document(request=request)
        return response
