import sys
import os
import glob

if len(sys.argv) != 2:
  print("Error: wrong format.\nUsage: python find_class.py [class_name]")
  sys.exit(0)

searching_class_name = sys.argv[1]

def find_class(class_name):
  file_list = glob.glob('*.txt')
  file_list.sort()
  # iterate through the text files
  file_found = False
  for txt_file in file_list:
    # open txt file lines to a list
    with open(txt_file) as f:
      content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    # go through each line of eache file
    for line in content:
      class_name = line.split()[0]
      if class_name == searching_class_name:
        print(" " + txt_file)
        file_found = True
        break
  if not file_found:
    print(" No file found with that class")

print("Ground-Truth folder:")
os.chdir("../ground-truth")
find_class(searching_class_name)
print("\nPredicted folder:")
os.chdir("../predicted")
find_class(searching_class_name)
