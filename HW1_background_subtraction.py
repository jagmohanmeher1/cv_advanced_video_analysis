import numpy as np
import cv2
    
def manual_subtraction(path):

    cap=cv2.VideoCapture("Dataset_CvDl_Hw2/Q1_Image/traffic.mp4")
    print(path)
    frames = []
    lock=1
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
   
    while cap.isOpened():
        ret, frame =cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(gray)
            if len(frames) < 25:
                frames.append(gray)
            else:
                if lock:
                    frames=np.array(frames)
                    mean=np.mean(frames,axis=0)   
                                             
                    standard_deviation=np.std(frames,axis=0) 
                    standard_deviation[standard_deviation<5]=5 
                    lock=0
                else:
                    mask[np.abs(gray - mean) > standard_deviation*5] = 255

            foreground = cv2.bitwise_and(frame, frame, mask= mask)
            mask_out = np.zeros_like(frame) 
            mask_out[:,:,0] = mask  
            mask_out[:,:,1] = mask
            mask_out[:,:,2] = mask

            outframes = np.hstack([frame, mask_out, foreground])

            cv2.imshow("Background subtraction", outframes)

            key = cv2.waitKey(10)

            if key == ord('q'):
                break
        else:
            return

    cap.release()
    cv2.destroyAllWindows()