import cv2
import numpy as np

# Model file paths
proto_file = "model\colorization_deploy_v2.prototxt"
model_file = "model\colorization_release_v2.caffemodel"
pts_file = "model\pts_in_hull.npy"

# Load cluster centers
pts_in_hull = np.load(pts_file)

# Load the network
net = cv2.dnn.readNetFromCaffe(proto_file, model_file)

# Add cluster centers to the model
pts = pts_in_hull.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

# Load black and white input image
image_path = "images\\1.jpg"   # change this to your file
frame = cv2.imread(image_path)

# Convert to LAB
frame_float = frame.astype("float32") / 255.0
lab = cv2.cvtColor(frame_float, cv2.COLOR_BGR2LAB)

L = lab[:, :, 0]

# Resize L channel to 224x224 for model
L_resized = cv2.resize(L, (224, 224))
L_resized -= 50

# Run model
net.setInput(cv2.dnn.blobFromImage(L_resized))
ab_decoded = net.forward()[0].transpose((1, 2, 0))

# Resize color channels back to original size
ab_resized = cv2.resize(ab_decoded, (frame.shape[1], frame.shape[0]))

# Combine with original L channel
lab_out = np.concatenate((L[:, :, np.newaxis], ab_resized), axis=2)

# Convert LAB back to BGR
colorized = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

# Save and show
output = (colorized * 255).astype("uint8")
cv2.imwrite("colorized_output.jpg", output)

cv2.imshow("Original", frame)
cv2.imshow("Colorized", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
