from ultralytics import YOLO
import cv2
from sort.sort import *
from util import get_car, write_csv, read_license_plate
results = {}

mot_tracker = Sort()

#load model
coco_model = YOLO("yolov8n.pt") #gonna use this model to detect cars
license_plate_detection = YOLO("Models\license_plate_detector.pt") #to detect number plates

#load video
cap = cv2.VideoCapture("video.mp4")

vehicle = [2,3,5,7] #obeject classids

#capture frames
frame_nmr = -1
ret = True
while ret :
    frame_nmr += 1
    ret, frame = cap.read()
    if ret and frame_nmr < 10:
        results[frame_nmr] = {}
        pass
    
        # detection of vehicle
        detections = coco_model(frame)[0]
        detections_ = []
        # print(detection)
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection 
            
            if int(class_id) in vehicle :
                detections_.append([x1,y1,x2,y2, score])
                
        #tracking vehicle 
        track_ids = mot_tracker.update(np.asarray(detections_))  # this will create an additional column or parameter(carid/objectid) which will detect the car througout the video
        
        #detect the lisence plate
        lisence_plates = license_plate_detection(frame)[0]
        for lisence_plate in lisence_plates.boxes.data.tolist():
            x1 ,y1 ,x2 ,y2 ,score, class_id = lisence_plate
            
            #assigning lisence plates to its car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(lisence_plate, track_ids)
            
            
            if car_id != -1 :
                
                #cropping the lisence plate
                lisence_plate_crop = frame[int(y1):int(y2),int(x1):int(x2), :]
                
                #processing lisence plate cropped frame
                lisence_plate_crop_gray = cv2.cvtColor(lisence_plate_crop, cv2.COLOR_BGR2GRAY)
                _, lisence_plate_crop_thresh = cv2.threshold(lisence_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                
                # cv2.imshow("lisence_plate_crop",lisence_plate_crop)
                # cv2.imshow("lisence_plate_thresh", lisence_plate_crop_thresh)
                
                # cv2.waitKey(0)
                
                
                #read lisence plate number
                
                
                
                license_plate_text, license_plate_text_score = read_license_plate(lisence_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
                    
#write results 
write_csv(results, "test2.csv")
                
    