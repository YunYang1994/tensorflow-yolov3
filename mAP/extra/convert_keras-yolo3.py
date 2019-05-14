'''
ABOUT THIS SCRIPT:
Converts ground-truth from the annotation files
according to the https://github.com/qqwweee/keras-yolo3
or https://github.com/gustavovaliati/keras-yolo3 format.

And converts the predicitons from the annotation files
according to the https://github.com/gustavovaliati/keras-yolo3 format.
'''

import argparse
import datetime
import os

'''
Each time this script runs, it saves the output in a different path
controlled by the following folder suffix: annotation_version.
'''
annotation_version = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

ap = argparse.ArgumentParser()

ap.add_argument("-o", "--output_path",
                required=False,
                default='from_kerasyolo3/version_{}'.format(annotation_version),
                type=str,
                help="The dataset root path location.")
ap.add_argument("-r", "--gen_recursive",
                required=False,
                default=False,
                action="store_true",
                help="Define if the output txt files will be placed in a \
                recursive folder tree or to direct txt files.")
group = ap.add_mutually_exclusive_group(required=True)
group.add_argument('--gt',
    type=str,
    default=None,
    help="The annotation file that refers to ground-truth in (keras-yolo3 format)")
group.add_argument('--pred',
    type=str,
    default=None,
    help="The annotation file that refers to predictions in (keras-yolo3 format)")

ARGS = ap.parse_args()

with open('class_list.txt', 'r') as class_file:
    class_map = class_file.readlines()
print(class_map)
annotation_file = ARGS.gt if ARGS.gt else ARGS.pred

os.makedirs(ARGS.output_path, exist_ok=True)

with open(annotation_file, 'r') as annot_f:
    for annot in annot_f:
        annot = annot.split(' ')
        img_path = annot[0].strip()
        if ARGS.gen_recursive:
            annotation_dir_name = os.path.dirname(img_path)
            # remove the root path to enable to path.join.
            if annotation_dir_name.startswith('/'):
                annotation_dir_name = annotation_dir_name.replace('/', '', 1)
            destination_dir = os.path.join(ARGS.output_path, annotation_dir_name)
            os.makedirs(destination_dir, exist_ok=True)
            # replace .jpg with your image format.
            file_name = os.path.basename(img_path).replace('.jpg', '.txt')
            output_file_path = os.path.join(destination_dir, file_name)
        else:
            file_name = img_path.replace('.jpg', '.txt').replace('/', '__')
            output_file_path = os.path.join(ARGS.output_path, file_name)
            os.path.dirname(output_file_path)

        with open(output_file_path, 'w') as out_f:
            for bbox in annot[1:]:
                if ARGS.gt:
                    # Here we are dealing with ground-truth annotations
                    # <class_name> <left> <top> <right> <bottom> [<difficult>]
                    # todo: handle difficulty
                    x_min, y_min, x_max, y_max, class_id = list(map(float, bbox.split(',')))
                    out_box = '{} {} {} {} {}'.format(
                        class_map[int(class_id)].strip(), x_min, y_min, x_max, y_max)
                else:
                    # Here we are dealing with predictions annotations
                    # <class_name> <confidence> <left> <top> <right> <bottom>
                    x_min, y_min, x_max, y_max, class_id, score = list(map(float, bbox.split(',')))
                    out_box = '{} {} {} {} {} {}'.format(
                        class_map[int(class_id)].strip(), score,  x_min, y_min, x_max, y_max)

                out_f.write(out_box + "\n")
