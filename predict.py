import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from model.naimishnet import NaimishNet


# Load an image and convert it to BGR to RGB
image = cv2.imread('samples_to_predict/mona_lisa.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load haar cascade for detecting frontal faces
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# Run the face detector
faces = face_cascade.detectMultiScale(image, 1.2, 2)

# Detect device type: CPU or GPU
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

net = NaimishNet()
net.load_state_dict(torch.load('saved_models/mymodel.pt', map_location=device))

image_copy = np.copy(image_rgb)

w_padding = 50
h_padding = 50

# index for printing the image
i = 0

for (x, y, w, h) in faces:
    # Select the region of interest
    roi = image_copy[y - h_padding : y + h + h_padding, x - w_padding : x + w + w_padding]

    # Convert the face region from BGR to gray scale
    image_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    image_normalized = image_gray / 255.0

    # Rescale the detected face to be the expected square size for the CNN (224, 224)
    image_resized = cv2.resize(image_normalized, (224, 224))

    # Reshape the numpy imag shape (HxWxC) into a torch image shape(CxHxW)
    image_reshaped = image_resized.reshape(image_resized.shape[0], image_resized.shape[1], 1)
    image_transposed = image_reshaped.transpose((2, 0, 1))
    torch_image = torch.from_numpy(image_transposed)

    # Make facial keypoints prediction using trained model
    torch_image.unsqueeze_(0)
    torch_image = torch_image.type(torch.FloatTensor)

    predicted_key_pts = net(torch_image)
    predicted_key_pts = predicted_key_pts.detach().numpy()

    # Denormalize the data
    predicted_key_pts = predicted_key_pts * 50.0 + 100
    predicted_key_pts = predicted_key_pts.reshape(68, 2)

    # Mark the keypoints on the sample image
    for cnt in range(len(predicted_key_pts[:, 0])):
        cv2.circle(image_resized, (predicted_key_pts[cnt, 0], predicted_key_pts[cnt, 1]), 2, (0,0,255), -1)

    cv2.imshow('Image', image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    i += 1
