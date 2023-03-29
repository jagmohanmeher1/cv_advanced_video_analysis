import cv2 
import numpy as np

def Processing(path):
    print("a")
    cap=cv2.VideoCapture(path)
    _,frames=cap.read()
    old_gray=cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # detector=creat_detector()
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 30
    params.maxThreshold = 220
    # # Filter by Area.
    params.filterByArea = True
    params.minArea = 30 
    params.maxArea = 100 
    # Filter by Circularity
    params.filterByCircularity = True 
    params.minCircularity = 1 
    # # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 1 
    # # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.5 

    detector = cv2.SimpleBlobDetector_create(params)
    key_points = detector.detect(old_gray)    

    
    
    for item in key_points:
        cv2.rectangle(frames, (int(item.pt[0]) - 6, int(item.pt[1]) - 6), (int(item.pt[0]) + 6, int(item.pt[1]) + 6),
                    (0, 0, 255), 1)
        cv2.line(frames, (int(item.pt[0]), int(item.pt[1] - 6)), (int(item.pt[0]), int(item.pt[1] + 6)), (0, 0, 255), 1)
        cv2.line(frames, (int(item.pt[0]) - 6, int(item.pt[1])), (int(item.pt[0]) + 6, int(item.pt[1])), (0, 0, 255), 1)

    point = []
    for item in key_points:
        #print(item.pt)     
        point.append([[item.pt[0], item.pt[1]]])   
    point = np.array(point, np.float32)
    print(point )
    cv2.imshow("detect", frames)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video_tracking(path):  
    print('b')
    
    cap=cv2.VideoCapture(path)
    
    _,frames=cap.read()
    old_gray=cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # detector=creat_detector()
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200
    # # Filter by Area.
    params.filterByArea = True
    params.minArea = 40 
    params.maxArea = 90 
    # Filter by Circularity
    params.filterByCircularity = True  
    params.minCircularity = 0.8   
    # # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.9   
    # # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.4  

    detector = cv2.SimpleBlobDetector_create(params)
    key_points = detector.detect(old_gray)    
    
    for item in key_points:
        cv2.rectangle(frames, (int(item.pt[0]) - 6, int(item.pt[1]) - 6), (int(item.pt[0]) + 6, int(item.pt[1]) + 6),
                    (0, 0, 255), 1)
        cv2.line(frames, (int(item.pt[0]), int(item.pt[1] - 6)), (int(item.pt[0]), int(item.pt[1] + 6)), (0, 0, 255), 1)
        cv2.line(frames, (int(item.pt[0]) - 6, int(item.pt[1])), (int(item.pt[0]) + 6, int(item.pt[1])), (0, 0, 255), 1)

    point = []
    for item in key_points:
        #print(item.pt)    
        point.append([[item.pt[0], item.pt[1]]])    
    point = np.array(point, np.float32)
    print(point )
    
    _,frames=cap.read()
    old_gray=cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    mask= np.zeros_like(frames)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

   
    while True:
        ret, frame = cap.read()
        if ret:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

            new_point, st, __ = cv2.calcOpticalFlowPyrLK(old_gray, gray, point, None, **lk_params)

            if  new_point is not None:   
                good_new = new_point[st == 1]  
                good_old = point[st == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                n1, n2 = new.ravel()   
                o1, o2 = old.ravel()
                mask = cv2.line(mask, (int(n1), int(n2)), (int(o1), int(o2)), (0, 255, 255), 3)
                frame = cv2.circle(frame, (int(n1), int(n2)), 5, (0, 255, 255), -1)

            img = cv2.add(frame, mask)
            cv2.imshow('Optical flow', img)
            
            key = cv2.waitKey(10)
            if key == ord('q'):
                break

            old_gray = gray.copy()
            point = good_new.reshape(-1, 1, 2)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


    
