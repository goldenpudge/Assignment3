import jetson.inference
import jetson.utils
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, loadImage, saveImage
net = detectNet("ssd-mobilenet-v2", threshold=0.7)

input_path = "/home/nvidia/Desktop/cat.jpeg"
img = loadImage(input_path)
detections = net.Detect(img)
for detection in detections:
  print(detection)
output_path="cat_result_0.7.jpg"
saveImage(output_path, img)


