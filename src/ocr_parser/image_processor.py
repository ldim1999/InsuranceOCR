from PIL import Image
import fitz
import tempfile
import logging

log = logging.getLogger(__name__)


class ImageProcessor(object):

    @classmethod
    def get_bytes(cls, image_file_path):
        with open(image_file_path, "rb") as image_file:
            return image_file.read()

    @classmethod
    def extract_images(cls, image_file_path):
        if image_file_path.endswith(('.jpg', '.gif')):
            img = Image.open(image_file_path)
            yield img
        elif image_file_path.endswith('.pdf'):
            doc = fitz.open(image_file_path)
            for page_idx, page in enumerate(doc):  # iterate through the pages
                try:
                    pix = page.get_pixmap()  # render page to an image
                    with tempfile.NamedTemporaryFile(delete=True, suffix='.png', mode='w') as wstream:
                        pix.save(wstream.name)
                        img = Image.open(wstream.name)
                        yield img
                except:
                    log.exception(f'Unable extract page {page_idx + 1} from {image_file_path}')
