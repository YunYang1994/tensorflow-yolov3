import os
import re

IN_FILE = 'result.txt'
OUTPUT_DIR = os.path.join('..', 'predicted')

SEPARATOR_KEY = 'Enter Image Path:'
IMG_FORMAT = '.jpg'

outfile = None
with open(IN_FILE) as infile:
    for line in infile:
        if SEPARATOR_KEY in line:
            if IMG_FORMAT not in line:
                break
            # get text between two substrings (SEPARATOR_KEY and IMG_FORMAT)
            image_path = re.search(SEPARATOR_KEY + '(.*)' + IMG_FORMAT, line)
            # get the image name (the final component of a image_path)
            # e.g., from 'data/horses_1' to 'horses_1'
            image_name = os.path.basename(image_path.group(1))
            # close the previous file
            if outfile is not None:
                outfile.close()
            # open a new file
            outfile = open(os.path.join(OUTPUT_DIR, image_name + '.txt'), 'w')
        elif outfile is not None:
            # split line on first occurrence of the character ':' and '%'
            class_name, info = line.split(':', 1)
            confidence, bbox = info.split('%', 1)
            # get all the coordinates of the bounding box
            bbox = bbox.replace(')','') # remove the character ')'
            # go through each of the parts of the string and check if it is a digit
            left, top, width, height = [int(s) for s in bbox.split() if s.lstrip('-').isdigit()]
            right = left + width
            bottom = top + height
            outfile.write("{} {} {} {} {} {}\n".format(class_name, float(confidence)/100, left, top, right, bottom))
