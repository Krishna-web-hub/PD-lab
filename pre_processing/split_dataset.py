import os
import shutil
import random

base = r"D:\PD_LAB\dataset"

splits = ["train", "val", "test"]

ratio = [0.7, 0.15, 0.15]

for task in os.listdir(base):

    task_path = os.path.join(base, task)

    if not os.path.isdir(task_path):
        continue

    for class_name in os.listdir(task_path):

        class_path = os.path.join(task_path, class_name)

        images = os.listdir(class_path)

        random.shuffle(images)

        total = len(images)

        train_end = int(total * ratio[0])
        val_end = train_end + int(total * ratio[1])

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        for img in train_imgs:
            dest = os.path.join(base, "train", task, class_name)
            os.makedirs(dest, exist_ok=True)
            shutil.copy(os.path.join(class_path, img), os.path.join(dest, img))

        for img in val_imgs:
            dest = os.path.join(base, "val", task, class_name)
            os.makedirs(dest, exist_ok=True)
            shutil.copy(os.path.join(class_path, img), os.path.join(dest, img))

        for img in test_imgs:
            dest = os.path.join(base, "test", task, class_name)
            os.makedirs(dest, exist_ok=True)
            shutil.copy(os.path.join(class_path, img), os.path.join(dest, img))

print("Dataset splitting completed.")