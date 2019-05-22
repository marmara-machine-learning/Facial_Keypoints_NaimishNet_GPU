import os

from torch.utils.data import Dataset, DataLoader
import matplotlib.image as mpimg # Module for basic image loading, rescaling etc.
import pandas as pd


class FacialKeyPointsDataset(Dataset):
    """ Facial keypoints dataset class."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        args:
            csv_file (string) : Path to csv file.
            root_dir (string) : Directory with all the images.
            transform (callable, optional) : Optional tranform to be applied on a sample.
        """

        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.key_pts_frame)


    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.key_pts_frame.iloc[idx, 0])

        image = mpimg.imread(image_name)

        # Delete the alpha channel if image has one(transparency)
        if (image.shape[2] == 4):
            image = image[:,:,0:3]

        key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        key_pts = key_pts.astype('float').reshape(-1,2)
        sample = {'image' : image, 'keypoints' : key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
