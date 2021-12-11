# information of data to be used, partition in test, train and validation by filename and idx

sub_folder = "/Users/julia/Documents/UNI/Master/Montréal/AV/project/duckietown_visual_odometry/data/"

train = {"sub_folder": sub_folder,
         "filenames": ['alex_2small_8_retest_ground_truth.txt',
                       'alex_2small_loops_ground_truth.txt',
                       'alex_3small_loops_ground_truth.txt',
                       'alex_train_complex_2_ground_truth.txt',
                       'razor_1small_8_ground_truth.txt',
                       'razor_2big_loops_ground_truth.txt',
                       'razor_2x3small_loops_ground_truth.txt'],
         "dir": ['alex_2small_8_retest_images',
                 'alex_2small_loops_images',
                 'alex_3small_loops_images',
                 'alex_train_complex_2_images',
                 'razor_1small_8_images',
                 'razor_2big_loops_images',
                 'razor_2x3small_loops_images'],
        "idx": [[200, 854], [0, 445], [200, 985], [0, 646], [100, 317], [0, 884], [200, 1297]]}

val = {"sub_folder": sub_folder,
       "filenames": ['alex_2small_8_retest_ground_truth.txt',
                     'alex_2small_loops_ground_truth.txt',
                     'alex_3small_loops_ground_truth.txt',
                     'alex_train_complex_2_ground_truth.txt',
                     'razor_1small_8_ground_truth.txt',
                     'razor_2big_loops_ground_truth.txt',
                     'razor_2x3small_loops_ground_truth.txt'],
       "dir": ['alex_2small_8_retest_images',
               'alex_2small_loops_images',
               'alex_3small_loops_images',
               'alex_train_complex_2_images',
               'razor_1small_8_images',
               'razor_2big_loops_images',
               'razor_2x3small_loops_images'],
       "idx": [[0, 200], [445, 644], [0, 200], [646, 845], [0, 100], [884, 1083], [0, 200]]}

test = {"sub_folder": sub_folder,
        "filenames": ['alex_test_complex_2_ground_truth.txt',
                      'razor_test_incomplet_ground_truth.txt'],
        "dir": ['alex_test_complex_2_images',
                'razor_test_incomplet_images'],
        "idx": [[0, 783], [0, 213]]}

train_dummy = {
    "sub_folder": sub_folder,
    "filenames": ["alex_2small_loops_ground_truth.txt"],
    "dir": ["alex_2small_loops_images"],
    "idx": [[0, 99]]
}

val_dummy = {
    "sub_folder": sub_folder,
    "filenames": ["alex_2small_loops_ground_truth.txt"],
    "dir": ["alex_2small_loops_images"],
    "idx": [[300, 400]]
}

test_dummy = {
    "sub_folder": sub_folder,
    "filenames": ["alex_2small_loops_ground_truth.txt"],
    "dir": ["alex_2small_loops_images"],
    "idx": [[400, 499]]
}
