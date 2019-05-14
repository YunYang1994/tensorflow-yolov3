import sys
import os
import glob
import xml.etree.ElementTree as ET


# change directory to the one with the files to be changed
path_to_folder = '../ground-truth'
#print(path_to_folder)
os.chdir(path_to_folder)

# old files (xml format) will be moved to a "backup" folder
## create the backup dir if it doesn't exist already
if not os.path.exists("backup"):
  os.makedirs("backup")

# create VOC format files
xml_list = glob.glob('*.xml')
if len(xml_list) == 0:
  print("Error: no .xml files found in ground-truth")
  sys.exit()
for tmp_file in xml_list:
  #print(tmp_file)
  # 1. create new file (VOC format)
  with open(tmp_file.replace(".xml", ".txt"), "a") as new_f:
    root = ET.parse(tmp_file).getroot()
    for obj in root.findall('object'):
      obj_name = obj.find('name').text
      bndbox = obj.find('bndbox')
      left = bndbox.find('xmin').text
      top = bndbox.find('ymin').text
      right = bndbox.find('xmax').text
      bottom = bndbox.find('ymax').text
      new_f.write(obj_name + " " + left + " " + top + " " + right + " " + bottom + '\n')
  # 2. move old file (xml format) to backup
  os.rename(tmp_file, "backup/" + tmp_file)
print("Conversion completed!")
