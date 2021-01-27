import cv2


cam_cap = cv2.VideoCapture(0)

while True:
    _, frame = cam_cap.read()

    cv2.imshow('self cam', frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_cap.release()
cv2.destroyAllWindows()
