import cv2

terminate_key = ord('q')  # Press 'q' to terminate the program

thres = 0.45  # Threshold to detect object

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    img = cv2.imread("image.png", cv2.IMREAD_COLOR)
    img = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))  # Resize image to half of its original size

    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=1)  # Changed thickness to 1
        cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Output", img)
    
    key = cv2.waitKey(1)
    if key == terminate_key:
        break

cv2.destroyAllWindows()
