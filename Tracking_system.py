#CAR AND PEDESTRAIN TRACKING SYSTEM
'''
Before using openCV use have to install openCV
cmd for installing openCV: pip install opencv-python
'''
import cv2
#importing video form the directory
video = cv2.VideoCapture("Pedestrians and cars.mp4")# Give the file name that you have in the directory
# Giving the trained cars data to cascade classifier
trained_car_data = cv2.CascadeClassifier('car_detector.xml')
# Giving the trained pedestrains data to cascade calssifier
pedestrain_data = cv2.CascadeClassifier('haarcascade_fullbody.xml')
# Run the video frame by frame
while True:
    (read_successful, frame) = video.read()
    # If the video is read successfully then convert the video to a black and white video
    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    # Find's the coordinate of cars using the trained cars data
    car_coordinate = trained_car_data.detectMultiScale(grayscaled_frame)
    # Find's the coordinate of the pedestrains using trained pedestrains data
    pedestrain_coordinate = pedestrain_data.detectMultiScale(grayscaled_frame)
    # Draw rectangle around the cars 
    for (x,y,w,h) in car_coordinate:
        cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 255, 0), 2)
    # Draw rectangle around the pedestrain
    for (x,y,w,h) in pedestrain_coordinate:
        cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 255, 255), 2)
    # Display the video
    cv2.imshow('car-Detector', frame)
    #
    key = cv2.waitKey(1)
    # Press Q to quit
    if key == 81 or key == 113:
        break
# Close the video stream
video.release()

print("code complete")
