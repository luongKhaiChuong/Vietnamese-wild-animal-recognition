import os
import cv2
from yolo_processor import Processor
from CLIP import ZeroShot

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
yolo = Processor()
clip_checker = ZeroShot(saved_model_file_name = "fin_model(1).pt")
proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections, out_frame = yolo.run(frame)
    is_rare = 0
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = det["crop"]
        results = clip_checker.classify_image(crop)
        print (results)
        top_res = results[0]
        top_animal, top_score = top_res["label"], top_res["confidence"]
        top_score = float(top_score)
        print (results, "top_score: ", top_score)
        if top_score > 0.4:
            is_rare = 1
            text = top_animal
            color = (0, 0, 255)
            print ("Rare ass animal ", text )
            #cv2.imshow("Heatmap", out_frame)
        else:
            print ("Normal ahh human")
            text = "Normal"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if (is_rare==1):
        cv2.imshow("Heatmap", out_frame)
    else:
        cv2.imshow("Frame", frame)
    if cv2.waitKey(100) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()