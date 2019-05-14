import sys
import os
import glob
import argparse

parser = argparse.ArgumentParser()
# argparse current class name to a list (since it can contain more than one word, e.g."dining table")
parser.add_argument('-c', '--current-class-name', nargs='+', type=str, help="current class name e.g.:\"dining table\".", required=True)
# new class name (should be a single string without any spaces, e.g. "diningtable")
parser.add_argument('-n', '--new-class-name', type=str, help="new class name.", required=True)
args = parser.parse_args()

current_class_name = " ".join(args.current_class_name) # join current name to single string
new_class_name = args.new_class_name


def query_yes_no(question, default="yes"):
  """Ask a yes/no question via raw_input() and return their answer.

  "question" is a string that is presented to the user.
  "default" is the presumed answer if the user just hits <Enter>.
      It must be "yes" (the default), "no" or None (meaning
      an answer is required of the user).

  The "answer" return value is True for "yes" or False for "no".
  """
  valid = {"yes": True, "y": True, "ye": True,
           "no": False, "n": False}
  if default is None:
    prompt = " [y/n] "
  elif default == "yes":
    prompt = " [Y/n] "
  elif default == "no":
    prompt = " [y/N] "
  else:
    raise ValueError("invalid default answer: '%s'" % default)

  while True:
    sys.stdout.write(question + prompt)
    if sys.version_info[0] == 3:
      choice = input().lower() # if version 3 of Python
    else:
      choice = raw_input().lower()
    if default is not None and choice == '':
      return valid[default]
    elif choice in valid:
      return valid[choice]
    else:
      sys.stdout.write("Please respond with 'yes' or 'no' "
                         "(or 'y' or 'n').\n")


def rename_class(current_class_name, new_class_name):
  # get list of txt files
  file_list = glob.glob('*.txt')
  file_list.sort()
  # iterate through the txt files
  for txt_file in file_list:
    class_found = False
    # open txt file lines to a list
    with open(txt_file) as f:
      content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    new_content = []
    # go through each line of eache file
    for line in content:
      #class_name = line.split()[0]
      if current_class_name in line:
        class_found = True
        line = line.replace(current_class_name, new_class_name)
      new_content.append(line)
    if class_found:
      # rewrite file
      with open(txt_file, 'w') as new_f:
        for line in new_content:
          new_f.write("%s\n" % line)

y_n_message = ("Are you sure you want "
               "to rename the class "
               "\"" + current_class_name + "\" "
               "into \"" + new_class_name + "\"?"
              )

if query_yes_no(y_n_message):
  print(" Ground-Truth folder:")
  os.chdir("../ground-truth")
  rename_class(current_class_name, new_class_name)
  print("  Done!")
  print(" Predicted folder:")
  os.chdir("../predicted")
  rename_class(current_class_name, new_class_name)
  print("  Done!")
