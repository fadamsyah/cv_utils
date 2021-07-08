from cv_utils.object_detection.dataset.splitter import split_coco

split_dictionary = {'train': 0.6,
                    'val': 0.2,
                    'test': 0.2,}

split_coco('test/ki67/instances.json', split_dictionary,
           'test/ki67', 2000)
