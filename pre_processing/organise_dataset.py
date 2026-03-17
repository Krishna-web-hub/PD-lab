import os
import shutil

base = r"D:\PD_LAB"
source = os.path.join(base, "Images")

for root, dirs, files in os.walk(source):

    for file in files:

        path = os.path.join(root, file)

        # AI generated images
        if "AI_gen_food" in root:

            dest = os.path.join(base, "dataset", "ai_detection", "ai_generated")
            os.makedirs(dest, exist_ok=True)

            shutil.copy(path, os.path.join(dest, file))


        # Good food images
        elif "good_food" in root:

            # copy for AI detection (real food)
            dest1 = os.path.join(base, "dataset", "ai_detection", "real_food")
            os.makedirs(dest1, exist_ok=True)
            shutil.copy(path, os.path.join(dest1, file))

            # copy for spoilage detection (fresh)
            dest2 = os.path.join(base, "dataset", "spoilage_detection", "fresh")
            os.makedirs(dest2, exist_ok=True)
            shutil.copy(path, os.path.join(dest2, file))


        # Bad food images
        elif "bad_food" in root:

            dest = os.path.join(base, "dataset", "spoilage_detection", "spoiled")
            os.makedirs(dest, exist_ok=True)

            shutil.copy(path, os.path.join(dest, file))


print("Dataset organization completed.")