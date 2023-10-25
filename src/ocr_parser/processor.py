import cv2
import fitz
import tempfile
import logging

log = logging.getLogger(__name__)

def extract_images(filename):
    if filename.endswith(('.jpg', '.gif')):
        img = cv2.imread(filename)
        yield img
    elif filename.endswith('.pdf'):
        doc = fitz.open(filename)
        for page_idx, page in enumerate(doc):  # iterate through the pages
            try:
                pix = page.get_pixmap()  # render page to an image
                with tempfile.NamedTemporaryFile(delete=True, suffix='.png', mode='w') as wstream:
                    pix.save(wstream.name)
                    img = cv2.imread(wstream.name)
                    yield img
            except:
                log.exception(f'Unable extract page {page_idx + 1} from {filename}')
