from re import S
import shutil
import os
import random

class system_split():
    def __init__(self):
        self.dataset_path = './CIFAR-10/'
        self.train_path = os.path.join(self.dataset_path, 'train/')
        self.val_path = os.path.join(self.dataset_path, 'val/')

        # Percentage of training data
        self.split_pc = 0.90

        self.create_directory(self.val_path)
        self.reset(self.val_path, self.train_path)
        self.split(self.train_path)

    def create_directory(self, path):
        """ Create directory """
        if not os.path.exists(path):
            os.makedirs(path)

    def get_file_ids(self, path):
        """ Get all fileids of a given folder path """
        ids = []
        for filename in os.listdir(path):
            ids.append(filename)
        return ids

    def reset(self, srcpath, dstpath):
        """ Returns val dataset back to train """
        for classname in os.listdir(srcpath):
            classpath = os.path.join(srcpath, classname)
            for filename in os.listdir(classpath):
                src = os.path.join(classpath, filename)
                dst = os.path.join(self.train_path, classname)
                shutil.move(src, dst)
    
    def split(self, path):
        """ Split train dataset into train + val"""
        for classname in os.listdir(path):
            classpath = os.path.join(path, classname)

            total_ids = self.get_file_ids(classpath)
            total_len = len(total_ids)
            val_len = round((1 - self.split_pc) * total_len)
            val_ids = random.sample(total_ids, val_len)

            dst = os.path.join(self.val_path, classname)
            self.create_directory(dst)

            for id in val_ids:
                src = os.path.join(classpath, id)
                shutil.move(src, dst)


system_split()
