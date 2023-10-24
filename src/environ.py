import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s [%(processName)s/%(threadName)s]: %(message)s')


def get_root_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


os.environ['TESSDATA_PREFIX'] = os.environ.get('TESSDATA_PREFIX', os.path.join(get_root_dir(), 'data', 'tessdata'))
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.environ.get(
    'GOOGLE_APPLICATION_CREDENTIALS', os.path.join(get_root_dir(), 'credentials', 'gcv.json'))
