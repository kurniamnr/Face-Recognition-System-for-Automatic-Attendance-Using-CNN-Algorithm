import cv2
import os

def rekamWajah(parameters):
    nama, nim, kelas = parameters
    namafolder = str(nim) +'_'+str(nama) + '_' + str(kelas)
    path = "dataset/"+namafolder
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    cap = cv2.VideoCapture(0)
    num_sample = 50
    i = 0
    while cap.isOpened():
        
        ret, frame = cap.read()
        if ret :
            namafile = str(nim) +"_"+str(nama) + "_" + str(kelas) + "_"+str(i) +'.jpg'
            cv2.imshow("Capture Photo", frame)
            cv2.imwrite(path + '/' + namafile, cv2.resize(frame, (250,250)))
            if cv2.waitKey(100) == ord('q') or i == num_sample:
                break
            i += 1    
    cap.release()
    cv2.destroyAllWindows()
    