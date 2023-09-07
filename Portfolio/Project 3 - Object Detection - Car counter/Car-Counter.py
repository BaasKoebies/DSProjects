from ultralytics import YOLO
import cv2 
# this package will allow us to display the detections
import cvzone
# importing package to track information
from sort import *

# this is activating the video based on path location
cap = cv2.VideoCapture("./Videos/cars.mp4")

# Can not set size of the video as with the webcam
# initialise the YOLO algorithim with medium weights
model = YOLO("../Yolo-Weights/yolov8m.pt")

# Outline the class names that will be used for classification of the detected objects
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# import the mask so that it can detect only in the region
mask = cv2.imread("./Project 1 - Car counter/mask.png")


# create instance for the tracker
# maximum age used to determine how many frames need to pass before this object is no longer tracked
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


# create a line that ensures that once the id object crosses it is it is counted (x1,y1,x2,y2)
#limits = [200, 420, 700, 420]
limits = [400, 297, 673, 297]

# number of cars total
total_cars = []

# while the webcam is still being used continue doing the following
while True:

    success, frame = cap.read()

    if not success:
        break
    
    # are going to overlay the mask and the frame together using a bit-wise operation of "AND" resulting in the 
    # desired region for detection
    frameReg = cv2.bitwise_and(frame, mask)
    
    
    # inserting graphic to be used in the analysis
    frameGraphics = cv2.imread("./Project 1 - Car counter/graphics.png", cv2.IMREAD_UNCHANGED)
    
    # overlay the graphic with the frame
    frame = cvzone.overlayPNG(frame, frameGraphics, (0, 0))
    
    
    # log the results of the model, based on the frame extracted from my webcam.
    # ensures efficient uses algorithm with the stream=True condition
    results = model(frameReg, stream=True)

    #create a list of the detections
    detections = np.empty((0,5))

    # this process checks for individuals bounding boxes for each of the results
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            # there is another format derived using the following command "box.xywh" which is the x, y, width and height
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 3)

            wdh, hgt = abs(x2-x1), abs(y2-y1)
            bbox = int(x1),int(y1),int(wdh),int(hgt)
            

            # Confidence 
            # determine the confidence of implementing the object detected
            conf = round(float(box.conf[0]),2)

            # Class name
            cls = int(box.cls[0])

            # creating scenario where I can specify what I want to detect 
            currentClass = classNames[cls]

            if (currentClass == "car" or currentClass == "motorbike" or currentClass == "bus" or currentClass ==  "truck") and conf>0.4:

                # displaying the detection classification on the image
                # max condition ensures that when the detected object 
                # moves out of the screen the confidence value can still 
                # be seen for the remaining portion of the object detected in the image


                # Save it to the detection list or detection array 
                currentArray = np.array((x1,y1,x2,y2,conf))
                
                # add the row of the new array onto the exist rows of the previous detections 
                detections = np.vstack((detections, currentArray) )

    # update tracker with a list of detections
    # attain the results for the tracker
    resultsTracker = tracker.update(detections)
    
    # draw the line on to the frame
    cv2.line(frame, (limits[0], limits[1]),(limits[2], limits[3]), color=(0,0,255), thickness=5)



    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        wdh, hgt = abs(x2-x1), abs(y2-y1)
        bbox2 = (x1,y1,wdh,hgt)
        cvzone.cornerRect(frame, bbox2, l=9, rt=2, colorR=(0,255,0))
        cvzone.putTextRect(frame, f' {int(id)}', (max(0,x1), max(30,y1)), scale=2, thickness=3, offset=10)


        # the condition is if the center of the detection crosses over the line it will increase the count
        cx, cy = x1+int(0.5*wdh),y1+int(0.5*hgt)
        cv2.circle(frame, (cx,cy), 2, color = (255,0,0), thickness=cv2.FILLED)

        # condition, depending on the speed the exact y-pixel may not be hit, hence a range was created to 
        # ensure the cy is not missed when it passes line
        if limits[0]<cx<limits[2] and limits[1]-20<cy<limits[3]+20:
            if id not in total_cars:
                total_cars.append(id)
                cv2.line(frame, (limits[0], limits[1]),(limits[2], limits[3]), color=(0,255,0), thickness=5)

    # Show the the total count function
    cv2.putText(frame,text=str(len(total_cars)),org=(255,100),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=5,color=(50,50,255),thickness =8)

    cv2.imshow("Frame", frame)
    #cv2.imshow("Region", frameReg)
    cv2.waitKey(1)