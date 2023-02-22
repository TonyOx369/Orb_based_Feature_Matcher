import numpy as np
import cv2
import time


path = r'C:\Users\admin\OneDrive\Desktop\Dreadnought\TAC Norway 2023\Aruco Markers\Vid1.mp4'
cap = cv2.VideoCapture(path)
cap2 = cv2.VideoCapture(path)

cap2.set(cv2.CAP_PROP_POS_FRAMES, 50)

prev_frame_time = 0
new_frame_time = 0
    
while cap.isOpened():

    ret, query_img = cap.read()
    query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
    query_img_bw = cv2.resize(query_img_bw, (query_img_bw.shape[0]*2, query_img_bw.shape[1]*2), interpolation=cv2.INTER_CUBIC)

    ret, train_img = cap2.read()
    train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.resize(train_img_bw, (train_img_bw.shape[0]*2, train_img_bw.shape[1]*2), interpolation=cv2.INTER_CUBIC)

    orb = cv2.ORB_create()

    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)

    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    # matcher = cv2.BFMatcher() #Brute Force Matcher
    matches = matcher.match(queryDescriptors,trainDescriptors)
    
    final_img = cv2.drawMatches(query_img_bw, queryKeypoints, train_img_bw, trainKeypoints, matches,None)

    font = cv2.FONT_HERSHEY_SIMPLEX

    new_frame_time = time.time()
# Calculating the FPS
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)

    h, w, _ = final_img.shape

    width = 1000
    height = int(width*(h/w))
    final_img = cv2.resize(final_img, (width, height*2), interpolation=cv2.INTER_CUBIC)

    cv2.putText(final_img, fps, (7, 70), font, 3, (100, 255, 0), 4, cv2.LINE_AA)

    cv2.imshow("Matches", final_img)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()


