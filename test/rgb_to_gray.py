import os
import sys
import cv2
import numpy as np 
from glob import glob
from tqdm import tqdm

rgb_path = sys.argv[1]
rgb_files = sorted(glob(os.path.join(rgb_path, "*.png")))

for i, rgb_file in enumerate(tqdm(rgb_files)):
    rgb = cv2.imread(rgb_file)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(rgb_file, gray)