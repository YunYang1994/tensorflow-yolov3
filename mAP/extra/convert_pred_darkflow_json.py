import sys
import os
import glob
import json


# change directory to the one with the files to be changed
path_to_folder = '../predicted'
#print(path_to_folder)
os.chdir(path_to_folder)

# old files (darkflow json format) will be moved to a "backup" folder
## create the backup dir if it doesn't exist already
if not os.path.exists("backup"):
  os.makedirs("backup")

# create VOC format files
json_list = glob.glob('*.json')
if len(json_list) == 0:
  print("Error: no .json files found in predicted")
  sys.exit()
for tmp_file in json_list:
  #print(tmp_file)
  # 1. create new file (VOC format)
  with open(tmp_file.replace(".json", ".txt"), "a") as new_f:
    data = json.load(open(tmp_file))
    for obj in data:
      obj_name = obj['label']
      conf = obj['confidence']
      left = obj['topleft']['x']
      top = obj['topleft']['y']
      right = obj['bottomright']['x']
      bottom = obj['bottomright']['y']
      new_f.write(obj_name + " " + str(conf) + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom) + '\n')
  # 2. move old file (darkflow format) to backup
  os.rename(tmp_file, "backup/" + tmp_file)
print("Conversion completed!")
