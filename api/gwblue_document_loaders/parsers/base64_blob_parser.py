import base64
from PIL import Image
from io import BytesIO
from langchain_community.document_loaders.parsers.images import BaseImageBlobParser


class Base64BlobParser(BaseImageBlobParser):
    """
    Pillow supports the following file formats:
    - BMP (Windows and OS/2 bitmap files)
    - EPS (Encapsulated PostScript)
    - GIF
    - JPEG and JPEG 2000
    - PCX
    - PNG
    - TIFF

    The file format is the storage scheme for the raw image data on disk (or in memory).
    Each file format has its own way of storing image data.

    Pillow tries to open the underlying file using "decoders" for each known image file format.
    The image decoder is the plug-in that knows how to interpret a particular file format.
    When Pillow sees a file with a .png extension, for instance, it calls the PNG "decoder"
    to parse the contents, decompress or decode the bits, and load them into a standard in-memory
    structure.

    Once loaded, regardless of the file format and decoder used, it still produces one Image object.

    If Pillow has a decoder for the format, Image.open(...) returns a PIL.Image.Image object —
    regardless of whether the underlying image is PNG, JPEG, GIF, or something else.

    The image data is loaded into a PIL.Image.Image object with a specific mode. The image mode is
    the internal representation of the pixel data in memory. Supported modes:
    - "1": 1-bit pixels, black and white
    - "L": 8-bit pixels, black and white (luminance)
    - "P": 8-bit pixels, mapped to a palette (color lookup table)
    - "RGB": 3x8-bit, true color
    - "RGBA": 4x8-bit, true color with alpha channel
    - "CMYK": 4x8-bit color
    - "YCbCr": 3x8-bit color (used by JPEG, for instance)
    - "LAB": 3x8-bit Lab color
    - "HSV": 3x8-bit Hue, Saturation, Value
    - "I": 32-bit signed integer pixels
    - "F": 32-bit floating point pixels

    Given the multimodal embeddings only support JPG and PNG, re-encode the
    in-memory PIL.Image.Image object into a new PNG file stored in memory.
    """

    def _analyze_image(self, img: Image.Image) -> str:
        with BytesIO() as output:
            img.save(output, format="PNG")
            image_bytes = output.getvalue()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f'<img src="data:image/png;base64,{encoded}" alt="Extracted PNG" />'
