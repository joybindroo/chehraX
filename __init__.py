__version__ = '0.1'
__title__ = 'Eameo-faceswap-generator'
__description__ = 'Face swap meme generator'
__uri__ = 'https://twitter.com/EameoOk'
__doc__ = __description__ + ' <' + __uri__ + '>'

__author__ = 'Nicolas Metallo'
__email__ = 'nicolas.metallo@nyu.edu'

__license__ = 'MIT License'

import sys

# Install pre-requisites
try:
    print("Installing required modules.. This may take a few minutes.")
    pip install -r requirements.txt
except Error:
    sys.exit("There was an error install the required modules")
