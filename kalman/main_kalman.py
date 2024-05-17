"""Filtro de kaman para deteccion de personas."""

import cv2
import numpy as np
import torch
from sort import Sort

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)


def object_tracking():
    """Funcion de tracking de objetos."""
    # cap = cv2.VideoCapture(r"/home/alan/Ambiente/kalman/test.mp4")
    cap = cv2.VideoCapture(0)
    mot_tracker = Sort()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("here")
            break

        results = model(frame)
        results = results.xyxy[0].cpu().numpy()
        person_detections = results[results[:, 5] == 0]

        dets = []
        for *xyxy, conf, cls in person_detections:
            x1, y1, x2, y2 = map(int, xyxy)
            dets.append([x1, y1, x2, y2, conf])
        dets = np.array(dets)

        trackers = mot_tracker.update(dets)

        for d in trackers:
            x1, y1, x2, y2, track_id = map(int, d[:5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                str(track_id),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                thickness=2,
            )

        cv2.imshow("view", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    object_tracking()
