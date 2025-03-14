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

    def _below_semantic_size_threshold(self, img: Image.Image) -> bool:
        """
        HuggingFace Transformers library has a bug. It performs a preprocessing
        step on images, e.g. reading, resizing, converting color formats,
        normalizing pixel values (e.g. subtracting means and dividing by standard
        deviations). The preprocessed image is transformed into PyTorch tensors
        fed to the LLM.

        The issue arises in channel dimensions. In RGB, for example, there are
        three channel dimensions: red, blue, green. It can have a shape as so:
        (3, 200, 200). This is known as channel first because the C value (number
        of channels) is indexed first at 3. The format is (C, H, W), where H is
        Height and W is weight. There is also a corrersponding channel last where
        the C value is indexed in last position: (H, W, C). The transformers library
        is doing normalization on pixels values: pixel - statistical mu / standard
        deviation. If an image is small, it confuses an RGB for a Grayscale image.
        Grayscale images only have one dimension.

        For normal images like 200x200x3, it is clear it is channel last and thus
        RGP. But for very small iamges like 1x1x3, is it channel first or channel
        last? If transformers infers channel first, then it treats it as a gray
        scale image. When it performs the standard deviation step, it expects one
        value for the normalization mean, but RGP has three means, e.g. [0.45, 0.43,
        0.406].

        Hence, to prevent the issue from arising, we rule out semantically
        insignificant images.

        TODO: consider classification model over this heuristic in future
        """
        return img.width < 250 or img.height < 250

    def _analyze_image(self, img: Image.Image) -> str:
        if self._below_semantic_size_threshold(img):
            return ""

        with BytesIO() as output:
            img.save(output, format="PNG")
            image_bytes = output.getvalue()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f'<img src="data:image/png;base64,{encoded}" alt="Extracted PNG" />'
