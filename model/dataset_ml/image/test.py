from PIL import Image
import numpy as np
from pathlib import Path

# test for gray image
img_gray = Image.open(Path(__file__).parent / "CE3_BMYK_PCAML-C-002_SCI_N_20140113190142_20140113190142_0008_A.png")
img_gray_array = np.array(img_gray).reshape([img_gray.width * img_gray.height, 3])
img_gray_array.reshape([img_gray.width, img_gray.height, 3])

# test for rgb image
img_rgb = Image.open(Path(
    __file__).parent.parent.parent.parent / "DataAnnotations" / "labelme" / "JPEGImages" / "CE3_BMYK_PCAML-C-002_SCI_N_20140113190142_20140113190142_0008_A.jpg")
img_rgb_array = np.array(img_rgb).reshape([img_rgb.width * img_rgb.height, 3])
img_rgb_array.reshape([img_rgb.width, img_rgb.height, 3])
