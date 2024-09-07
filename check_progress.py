# go through each subfolder under /lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/droid_with_pcd_20k and find how many have "mark.txt" file

import os
import glob
import json
with open("/lustre/fsw/portfolios/nvr/users/lawchen/project/droid/droid/aggregated-annotations-030724.json", 'r') as file:
    annotations = json.load(file)

def check_progress():
    folder = "/lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/droid_with_pcd_5k"
    subfolders = glob.glob(os.path.join(folder, "*"))
    print(f"Found {len(subfolders)} subfolders")
    counter = 0
    for subfolder in subfolders:
        # check if there is a mark.txt file
        mark_file = os.path.join(subfolder, "mark.txt")
        # if not os.path.exists(mark_file):
        #     # delete the subfolder
        #     print(f"Deleting {subfolder}")
        #     os.system(f"rm -rf {subfolder}")
        # delete mark.txt
        if os.path.exists(mark_file):
        #     os.system(f"rm {mark_file}")
            continue



        # find a file that starts with "metadata" and ends with ".json"
        # metadata_files = glob.glob(os.path.join(subfolder, "metadata*.json"))
        # if len(metadata_files) != 1:
        #     print(f"Found {len(metadata_files)} metadata files in {subfolder}")
        #     # remove the subfolder
        #     print(f"Deleting {subfolder}")
        #     os.system(f"rm -rf {subfolder}")
        #     continue
        # metadata_file = metadata_files[0]
        # # extract "TRI+52ca9b6a+2023-11-07-15h-30m-09s" from "metadata_TRI+52ca9b6a+2023-11-07-15h-30m-09s.json"
        # metadata_name = os.path.basename(metadata_file).split(".")[0]
        # metadata_name = metadata_name.split("_")[1]
        # # find the corresponding annotation
        # if metadata_name not in annotations:
        #     print(f"Missing annotation for {metadata_name}")
        #     # remove the subfolder
        #     print(f"Deleting {subfolder}")
        #     os.system(f"rm -rf {subfolder}")
        else:
            counter += 1
            # remove the subfolder
            print(f"Deleting {subfolder}")
            os.system(f"rm -rf {subfolder}")
    return counter

if __name__ == '__main__':
    print(check_progress())