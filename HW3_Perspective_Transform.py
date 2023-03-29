import cv2 
import numpy as np

def Perspective_Transfrom(path_vid,path_img):  
    video = cv2.VideoCapture("/Users/jagmohanmeher/Documents/NCKU/3rd sem/Computer vision course/HW2_final/Dataset_CvDl_Hw2/Q3_Image/video.mp4")
    img = cv2.imread("//Users/jagmohanmeher/Documents/NCKU/3rd sem/Computer vision course/HW2_final/Dataset_CvDl_Hw2/Q3_Image/logo.png")
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters_create()
    while True:
        ret, frame = video.read()
        original_frame = np.copy(frame)
        if ret:
            print('a') 
            corners, ids, rejected_img_points =cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
            print(corners)
            id1 = np.squeeze(np.where(ids == 1))
            id2 = np.squeeze(np.where(ids == 2))
            id3 = np.squeeze(np.where(ids == 3))
            id4 = np.squeeze(np.where(ids == 4))
            print(id1)
            print(id2)
            print(id3)
            print(id4)    
            if id1 != [] and id2 != [] and id3 != [] and id4 != []: 
                print('c')
                pt1 = np.squeeze(corners[id1[0]])[0]    
                pt2 = np.squeeze(corners[id2[0]])[1]   
                pt3 = np.squeeze(corners[id3[0]])[2]    
                pt4 = np.squeeze(corners[id4[0]])[3]   

                pts_dst = [[pt1[0], pt1[1]]]
                pts_dst = pts_dst + [[pt2[0], pt2[1]]]
                pts_dst = pts_dst + [[pt3[0], pt3[1]]]
                pts_dst = pts_dst + [[pt4[0], pt4[1]]]

                
                pts_src = [[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]]

                M, mask = cv2.findHomography(np.float32(pts_src), np.float32(pts_dst), cv2.RANSAC)

                h, w = frame.shape[:2]
                result = cv2.warpPerspective(img, M, (w,h))
                
                mask2 = np.zeros(frame.shape, dtype=np.uint8)

                roi_corners2 = np.int32(pts_dst)
                channel_count2 = frame.shape[2]  
                ignore_mask_color2 = (255,)*channel_count2

                cv2.fillConvexPoly(mask2, roi_corners2, ignore_mask_color2)

                mask2 = cv2.bitwise_not(mask2)
                masked_image2 = cv2.bitwise_and(frame, mask2)

                result = cv2.bitwise_or(result, masked_image2)

                result = np.hstack((original_frame, result))
                print('b')
                cv2.namedWindow('Perspective transformation', 0)
                cv2.imshow('Perspective transformation', result)
                key = cv2.waitKey(5) 
                if key == ord('q'):
                    break
        else:
            break





