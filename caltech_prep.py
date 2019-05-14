import random
import os
import shutil

data_dir = "/home/nisha/data/256_ObjectCategoriesSplit/train"
output_data_dir = "/home/nisha/data/256_ObjectCategoriesSubset"

if __name__ == '__main__':
    if not os.path.exists(output_data_dir):
        os.mkdir(output_data_dir)
        train_data_dir = os.path.join(output_data_dir, 'train')
        val_data_dir = os.path.join(output_data_dir, 'valid')
        os.mkdir(train_data_dir)
        os.mkdir(val_data_dir)
    
        # dirnames = os.listdir(data_dir)
        dirnames = ["026.cake", "087.goldfish", "239.washing-machine", "057.dolphin-101",
                    "170.rainbow", "241.waterfall", "064.elephant-101", "212.teapot", 
                    "086.golden-gate-bridge", "213.teddy-bear"]
        for dirname in dirnames:
            print(dirname)
            
            dirname_complete = os.path.join(data_dir, dirname)
            filenames = os.listdir(dirname_complete)
            filenames = [os.path.join(dirname_complete, f) for f in filenames if f.endswith('.jpg')]
            random.seed(123)
            filenames.sort()
            random.shuffle(filenames)

            split = int(0.8 * len(filenames))
            train_filenames = filenames[:split]
            # train_filenames = [os.path.join(dirname, f) for f in train_filenames if f.endswith('.jpg')]
            val_filenames = filenames[split:]
            # val_filenames = [os.path.join(dirname, f) for f in val_filenames if f.endswith('.jpg')]
            
            train_img_dirname = os.path.join(output_data_dir, 'train', dirname)
            os.mkdir(train_img_dirname)
            val_img_dirname = os.path.join(output_data_dir, 'valid', dirname)
            os.mkdir(val_img_dirname)
            
            for filename in val_filenames:
                # print filename
                shutil.copy2(filename, val_img_dirname)

            for filename in train_filenames:
                shutil.copy2(filename, train_img_dirname)
       
    else:
        print("Warning: output dir {} already exists".format(output_data_dir))
