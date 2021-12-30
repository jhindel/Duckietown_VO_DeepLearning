# information of data to be used, partition in test, train and validation by filename and idx

train = {"filenames": ['alex_2small_8_retest_ground_truth.txt',
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

val = {"filenames": ['alex_2small_8_retest_ground_truth.txt',
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

test = {"filenames": ['alex_test_complex_2_ground_truth.txt',
                      'razor_test_incomplet_ground_truth.txt'],
        "dir": ['alex_test_complex_2_images',
                'razor_test_incomplet_images'],
        "idx": [[0, 783], [0, 213]]}

train_dummy = {
    "filenames": ["alex_2small_loops_ground_truth.txt"],
    "dir": ["alex_2small_loops_images"],
    "idx": [[0, 445]]
}

val_dummy = {
    "filenames": ["alex_2small_loops_ground_truth.txt"],
    "dir": ["alex_2small_loops_images"],
    "idx": [[445, 644]]
}

test_dummy = {
    "filenames": ['alex_3small_loops_ground_truth.txt'],
    "dir": ['alex_3small_loops_images'],
    "idx": [[50, 250]]
}

train_alex = {"filenames": ['alex_2small_8_retest_ground_truth.txt',
                       'alex_2small_loops_ground_truth.txt',
                       'alex_3small_loops_ground_truth.txt',
                       'alex_train_complex_2_ground_truth.txt'],
         "dir": ['alex_2small_8_retest_images',
                 'alex_2small_loops_images',
                 'alex_3small_loops_images',
                 'alex_train_complex_2_images'],
        "idx": [[200, 854], [0, 445], [200, 985], [0, 646]]}

val_alex = {"filenames": ['alex_2small_8_retest_ground_truth.txt',
                     'alex_2small_loops_ground_truth.txt',
                     'alex_3small_loops_ground_truth.txt',
                     'alex_train_complex_2_ground_truth.txt'],
       "dir": ['alex_2small_8_retest_images',
               'alex_2small_loops_images',
               'alex_3small_loops_images',
               'alex_train_complex_2_images'],
       "idx": [[0, 200], [445, 644], [0, 200], [646, 845]]}

test_alex = {"filenames": ['alex_test_complex_2_ground_truth.txt'],
        "dir": ['alex_test_complex_2_images'],
        "idx": [[0, 783]]}
