git clone https://github.com/YunYang1994/raccoon_dataset.git
cat ./raccoon_dataset/labels.txt | head -n  180 > ./raccoon_dataset/train.txt
cat ./raccoon_dataset/labels.txt | tail -n +181 > ./raccoon_dataset/test.txt
python core/convert_tfrecord.py --dataset_txt ./raccoon_dataset/train.txt --tfrecord_path_prefix ./raccoon_dataset/raccoon_train
python core/convert_tfrecord.py --dataset_txt ./raccoon_dataset/test.txt  --tfrecord_path_prefix ./raccoon_dataset/raccoon_test
