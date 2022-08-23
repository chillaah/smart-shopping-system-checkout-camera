
import cv2
import numpy as np
import argparse
# import picamera
import time

# with picamera.PiCamera() as cam:
#     print(cam.MAX_RESOLUTION)
# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('-c', '--webcam', type=int, required=False, help='Enter webcam id', default=0)
# parser.add_argument('-v','--video', type=str, required=False,help='Enter video path')

# Parse the argument
args = parser.parse_args()

# Load Yolo
net = cv2.dnn.readNet("yolov4-tiny-custom_final.weights", "yolov4-tiny-custom.cfg")
# classes = ['100','90','80','70']
classes = ['rice', 'tea', 'honey', 'jam']
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# cap = cv2.VideoCapture('IMG_8975.MOV')
count = 0

vid = cv2.VideoCapture(args.webcam)
# vid = cv2.VideoCapture(-1)
prev_frame_time = 0
new_frame_time = 0
while True:
    label = '-'
    # while (cap.isOpened()):
    count += 1

    # Capture frame-by-frame
    # ret, frame = cap.read()
    ret, frame = vid.read()

    # if(count%5 != 0):
    #     continue
    # frame = cv2.resize(frame, (512, 512), fx = 0, fy = 0,
    #                      interpolation = cv2.INTER_CUBIC)

    # frame = cv2.resize(frame, (416,416), interpolation = cv2.INTER_AREA)
    # frame = cv2.reshape(frame,(416,416,3))
    height, width, channels = frame.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # frame rate calculation and display
    new_frame_time = time.time()
    fps_disp = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps_disp = int(fps_disp)
    fps_disp = str(fps_disp)
    cv2.putText(frame, fps_disp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # print(confidence)
            if confidence > 0.8:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.6)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

vid.release()
# cap.release()
cv2.destroyAllWindows()