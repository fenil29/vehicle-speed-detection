
# //////////////////////////////////////////////////////////////////////////////////////////////
# simultaneous
import threading
import numpy as np
import cv2
import time
import os

input_image = 'input-image.jpg'
input_video = 'input.mp4'
# Let's load a simple image with 3 black squares
image = cv2.imread(input_image)
cv2.waitKey(0)

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(gray, 30, 200)
cv2.waitKey(0)

# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
contours, hierarchy = cv2.findContours(edged,
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.imshow('Canny Edges After Contouring', edged)
# cv2.waitKey(0)
arr = []
arr2 = []

print("Number of Contours found = " + str(len(contours)))
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    # print(w)
    # if cv2.contourArea(contour) < 0:
    #     continue
    if w < 100:
        continue
    # if h > 10:
    #     continue
    # print(x, y)
    arr.append(y)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, f"speed:{x,y} km/h", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3)
# Draw all contours
# -1 signifies drawing all contours
# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
arr.sort()
for i in range(len(arr)-1):
    if arr[i+1]-arr[i] > 70:
        arr2.append(arr[i])
temp = abs(arr2[-1]-arr[-1])
if temp > 70:
    arr2.append(arr[-1])

for y in arr2:
    cv2.line(image, (100, y), (1000, y), (255, 0, 0), 5)

print(arr)
print(arr2)
cv2.imshow('Contours', image)
# cv2.waitKey(0)
cv2.destroyAllWindows()

# from ..convenience import is_cv3

# arr2 = [15, 172, 325, 487, 649, 803, 963]
cap = cv2.VideoCapture(input_video)
# cap = cv2.VideoCapture('5ct.mp4')
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(5)
frame_count = 0

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ret, frame1 = cap.read()
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = []
framet = []
if not os.path.exists(f"split"):
    os.makedirs(f"split")
for i in range(len(arr2)-1):
    out.append(0)
    out.append(0)

for i in range(len(arr2)-1):
    upper = arr2[i]
    lower = arr2[i+1]
    out[i] = cv2.VideoWriter(f"split\\output{i+1}.avi", fourcc, fps,
                             (frame1.shape[1], lower-upper))

while cap.isOpened():
    ret, frame = cap.read()
    # image = cv2.resize(frame1, (frame1.shape[1], frame1.shape[0]))
    # (x, y, w, h) = cv2.boundingRect(c)
    # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    try:
        for i in range(len(arr2)-1):
            upper = arr2[i]
            lower = arr2[i+1]
            try:
                framet = frame[upper:lower, 0:frame_width]
            except Exception:
                raise Exception('End')
            out[i].write(framet)
            cv2.imshow("feed", framet)
    except Exception:
        break
    # frame1 = frame2
    # print('FPS {:.1f}'.format(1 / (time.time() - stime)))

    if cv2.waitKey(40) == 27:
        break
    frame_count += 1
    print(f"{round((frame_count/total_frame)*100, 2)}% (1/2)")

cv2.destroyAllWindows()
cap.release()
for i in range(len(arr2)-1):
    out[i].release()

# /////////////////////////////////////////////////////////////////////////////////

count_video2 = 0
if not os.path.exists(f"final"):
    os.makedirs(f"final")
if not os.path.exists(f"final\\image"):
    os.makedirs(f"final\\image")
if not os.path.exists(f"final\\image\\overspeed"):
    os.makedirs(f"final\\image\\overspeed")


def calculate(num_video):

    global count_video2

    if not os.path.exists(f"final\\image\\{num_video}"):
        os.makedirs(f"final\\image\\{num_video}")

    cap = cv2.VideoCapture(f'split\\output{num_video}.avi')
    # cap = cv2.VideoCapture('5ct.mp4')
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(5)
    # sec_per_frame = (total_frame/fps) / total_frame
    sec_per_frame = 1 / fps
    print(sec_per_frame)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    ret, frame1 = cap.read()
    out = cv2.VideoWriter(f"final\\output{num_video}.avi", fourcc, fps,
                          (frame1.shape[1], frame1.shape[0]))
    # frame1.shape[0]->width of each lane in highway
    km_per_pix = 0.0035/frame1.shape[0]
    threshold_area_for_vehicle = (frame1.shape[0]*frame1.shape[0])*2/3
    print(threshold_area_for_vehicle)
    # out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1920, 160))

    # ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    print(frame1.shape)

    count = 0
    bool_for_taking_image_onetime = False
    count_frame_start = False
    count_frame_start_value = 0
    count_frame_end = False
    count_frame_end_value = 0
    countframe = 0
    count_time1 = 0
    count_frame1 = 0
    count_time2 = 0
    count_frame2 = 0
    while cap.isOpened():
        count_video2 += 1
        print(
            f"{(round(((count_video2/total_frame)*100)/(len(arr2)-1), 2))}% (2/2)")
        stime = time.time()
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            start_detection = 100
            stop_detection = 400
            speed_limit = 100
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.line(frame1, (stop_detection, 10),
                     (stop_detection, 100), (255, 0, 0), 5)
            cv2.line(frame1, (start_detection, 10),
                     (start_detection, 100), (255, 0, 0), 5)
            if cv2.contourArea(contour) < threshold_area_for_vehicle:
                continue
            if (x+w*0.5) > stop_detection and count == 2:
                # cv2.putText(frame1, f"speed:{(count_frame2-count_frame1)*sec_per_frame} km/h", (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                #             1, (0, 0, 255), 3)
                # cv2.putText(frame1, f"speed:{(  ((count_frame_end_value-count_frame_start_value)*km_per_pix) / (((count_frame2-count_frame1)*sec_per_frame)/3600) if  (((count_frame2-count_frame1)*sec_per_frame)/3600)!=0 else 1 )  } km/h", (int(x+w*0.5), int(y+h*0.5)), cv2.FONT_HERSHEY_SIMPLEX,
                #             1, (0, 0, 255), 3)
                cv2.putText(frame1, f"speed:{(  ((count_frame_end_value-count_frame_start_value)*km_per_pix) / (((count_frame2-count_frame1)*sec_per_frame)/3600) if  (((count_frame2-count_frame1)*sec_per_frame)/3600)!=0 else 1 )  } km/h", (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)

                if bool_for_taking_image_onetime and (stop_detection+100 > (x+w*0.5) > stop_detection):
                    bool_for_taking_image_onetime = False
                    cv2.rectangle(frame1, (x, y),
                                  (x+w, y+h), (255, 0, 0), 2)
                    if (((count_frame_end_value-count_frame_start_value)*km_per_pix) / (((count_frame2-count_frame1)*sec_per_frame)/3600) if (((count_frame2-count_frame1)*sec_per_frame)/3600) != 0 else 1) > speed_limit:
                        cv2.imwrite(
                            f"final\\image\\overspeed\\{(((count_frame_end_value-count_frame_start_value)*km_per_pix) / (((count_frame2-count_frame1)*sec_per_frame)/3600) if  (((count_frame2-count_frame1)*sec_per_frame)/3600)!=0 else 1 ) }image.jpg", frame1)

                    cv2.imwrite(
                        f"final\\image\\{num_video}\\{(((count_frame_end_value-count_frame_start_value)*km_per_pix) / (((count_frame2-count_frame1)*sec_per_frame)/3600) if  (((count_frame2-count_frame1)*sec_per_frame)/3600)!=0 else 1 ) }image.jpg", frame1)
            if x+w*0.5 > stop_detection:
                continue
            if x+w*0.5 < start_detection:
                continue

            if (x+w*0.5) < (start_detection+100) and not count_frame_start:
                count_frame_start = True
                count_frame_end = False
                count_time1 = time.time()
                count_frame1 = countframe
                count = 1
                count_frame_start_value = (x+w*0.5)
                # print((x+w*0.5))

            if (x+w*0.5) > (stop_detection-100) and not count_frame_end and count_frame_start:
                count_frame_end = True
                count_frame_start = False
                bool_for_taking_image_onetime = True

                count_time2 = time.time()
                count_frame2 = countframe
                count = 2
                count_frame_end_value = (x+w*0.5)
                # print((x+w*0.5))

            # print(cv2.contourArea(contour))
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (0, 0, 255), 3)
            # print(x+w*0.5, y+h*0.5)
            cv2.circle(frame1, (int(x+w*0.5), int(y+h*0.5)),
                       10, (0, 0, 255), 50)
        # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

        image = cv2.resize(frame1, (frame1.shape[1], frame1.shape[0]))
        out.write(image)
        cv2.imshow(f"feed{num_video}", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()
        countframe += 1
        # print('FPS {:.1f}'.format(1 / (time.time() - stime)))

        if cv2.waitKey(40) == 27:
            brea

    cv2.destroyAllWindows()
    cap.release()
    out.release()


t = []
for i in range(len(arr2)-1):
    t.append(threading.Thread(target=calculate, args=(i+1,)))

# # starting thread 1
# t1.start()
# # # starting thread 2
# t2.start()
# t3.start()
for i in t:
    i.start()


print("Done!")
