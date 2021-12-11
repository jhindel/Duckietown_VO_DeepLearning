from os import walk
import pandas as pd

# to figure out how much data is available and how to split it

all_files = {"sub_folder": "/Users/julia/Documents/UNI/Master/Montréal/AV/project/duckietown_visual_odometry/data/",
             "filenames": ['alex_2small_8_retest_ground_truth.txt',
                           'alex_2small_loops_ground_truth.txt',
                           'alex_3small_loops_ground_truth.txt',
                           'alex_test_complex_2_ground_truth.txt',
                           'alex_train_complex_2_ground_truth.txt',
                           'razor_1small_8_ground_truth.txt',
                           'razor_2big_loops_ground_truth.txt',
                           'razor_2x3small_loops_ground_truth.txt',
                           'razor_test_incomplet_ground_truth.txt'],
             "dir": ['alex_2small_8_retest_images',
                     'alex_2small_loops_images',
                     'alex_3small_loops_images',
                     'alex_test_complex_2_images',
                     'alex_train_complex_2_images',
                     'razor_1small_8_images',
                     'razor_2big_loops_images',
                     'razor_2x3small_loops_images',
                     'razor_test_incomplet_images']}

sum = 0
for i in range(len(all_files["filenames"])):
    # read txt file
    gt_file = pd.read_fwf(all_files["sub_folder"] + all_files["filenames"][i])
    # get list of all files in dir
    all_filenames_dir = sorted(next(walk(all_files["sub_folder"] + all_files["dir"][i]), (None, None, []))[2])
    print(i, all_files["filenames"][i], gt_file.shape, len(all_filenames_dir))
    sum += len(all_filenames_dir)

print("total no. of frames", sum)

print("Training:", 0.7*sum, "Validation", 0.2*sum, "Testing", 0.1*sum)

