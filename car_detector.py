import cv2
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from PIL import Image
from siec_neuro import car_or_track

try:
    yolo_model = YOLO('yolov8l.pt')
    print("Model YOLO załadowany.")
except Exception as e:
    print(f"Błąd wczytania modelu YOLO: {e}")
    exit()

model_car_or_track = car_or_track()
model_car_or_track.load_state_dict(torch.load('model.pth'))
model_car_or_track.eval()
print("Model car_or_track załadowany.")

video_path = "your_path"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Nie udało się otworzyć pliku wideo!")
    exit()

frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Parametry śledzenia
next_id = 0
objects = {}  
iou_threshold = 0.3  # Minimalna wartość IoU do przypisania ID
total_cars = 0  # Licznik samochodów

def iou_tensor(boxes1, boxes2):
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter
    return inter / union

# Pętla przetwarzania wideo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Koniec wideo lub błąd odczytu.")
        break

    # Detekcja obiektów YOLO
    results = yolo_model(frame, conf=0.3)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2, conf, cls = box.xyxy[0].int().tolist() + [box.conf[0].item(), box.cls[0].item()]
        if conf > 0.5 and yolo_model.names[int(cls)] in ['car', 'truck']: # conf = 0.5
            detections.append([x1, y1, x2, y2])

    detections = torch.tensor(detections, dtype=torch.float32) if detections else torch.empty((0, 4), dtype=torch.float32)

    updated_objects = {}
    if objects:
        existing_boxes = torch.stack([obj['box'] for obj in objects.values()])
        ious = iou_tensor(detections, existing_boxes)
        max_ious, indices = ious.max(dim=1)

        for i, (iou_val, idx) in enumerate(zip(max_ious, indices)):
            if iou_val > iou_threshold:
                obj_id = list(objects.keys())[idx.item()]
                updated_objects[obj_id] = {'box': detections[i], 'frames': 0, 'class': objects[obj_id]['class']}
            else:
                cropped_img = frame[int(detections[i][1]):int(detections[i][3]), int(detections[i][0]):int(detections[i][2])]
                if cropped_img.size > 0:
                    pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                    input_tensor = transform(pil_img).unsqueeze(0)
                    
                    # Predykcja modelu z PyTorcha
                    with torch.no_grad():
                        outputs = model_car_or_track(input_tensor)
                        pred_class = ["car", "truck"][torch.argmax(outputs, 1).item()]

                    updated_objects[next_id] = {'box': detections[i], 'frames': 0, 'class': pred_class}
                    if pred_class == "car":
                        total_cars += 1
                    next_id += 1
    else:
        for det in detections:
            cropped_img = frame[int(det[1]):int(det[3]), int(det[0]):int(det[2])]
            if cropped_img.size > 0:
                pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                input_tensor = transform(pil_img).unsqueeze(0)

                with torch.no_grad():
                    outputs = model_car_or_track(input_tensor)
                    pred_class = ["car", "truck"][torch.argmax(outputs, 1).item()]

                updated_objects[next_id] = {'box': det, 'frames': 0, 'class': pred_class}
                if pred_class == "car":
                    total_cars += 1
                next_id += 1

    # Aktualizacja obiektów
    for obj_id, data in objects.items():
        if obj_id not in updated_objects:
            data['frames'] += 1
            if data['frames'] < 5:
                updated_objects[obj_id] = data

    objects = updated_objects

    # Rysowanie prostokątów i wyświetlanie ID i klasy
    for obj_id, data in objects.items():
        x1, y1, x2, y2 = map(int, data['box'].tolist())
        label = f"ID: {obj_id}, {data['class']}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Wyświetlanie licznika aut
    cv2.putText(frame, f"Licznik aut: {total_cars}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Przerwano przez użytkownika.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
