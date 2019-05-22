import torch


class ToTensor(object):
    """ Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        # If image has no grayscale color channel, add one
        if len(image.shape) == 2:
            # Add to third color dimension
            image = image.reshape(image.shape[0], image.shape[1], 1)

        # Swap color axis
        # The reason:
        # numpy image : H x W x C
        # torch image : C x H x W
        image = image.transpose((2, 0, 1))

        return {'image' : torch.from_numpy(image), 'keypoints' : torch.from_numpy(key_pts)}
