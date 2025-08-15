import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import os
from datetime import datetime
import winsound  # Import winsound for audio alerts on Windows
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage  # Import MIMEImage for image attachments
import threading  # Import threading to handle concurrent tasks
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Gmail configuration
GMAIL_USER = 'paybox5779@gmail.com'
GMAIL_PASSWORD = 'wudb nelc qbes ciwt'  # Use your App Password if 2FA is enabled

def send_email_alert(vehicle_id, timestamp, image_path=None):
    subject = f'Wrong Way Vehicle Detected: {vehicle_id}'
    body = f'A vehicle with ID {vehicle_id} was detected going the wrong way at {timestamp}.'

    msg = MIMEMultipart()
    msg['From'] = GMAIL_USER
    msg['To'] = GMAIL_USER  # Send to yourself
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    # Attach the image if provided
    if image_path:
        with open(image_path, 'rb') as image_file:
            msg.attach(MIMEImage(image_file.read(), name=os.path.basename(image_path)))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()  # Use TLS
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.send_message(msg)
            print("Email alert sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

def play_alert_sound():
    winsound.PlaySound(alert_sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture("wrongway.mp4")
with open("coco.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

# Define the areas
area1 = [(593, 227), (602, 279), (785, 274), (774, 220)]
area2 = [(747, 92), (785, 208), (823, 702), (773, 95)]
save_dir = "wrong_way_cars"
os.makedirs(save_dir, exist_ok=True)
print(f"Saving images to directory: {save_dir}")

# Path to the alert sound (use .wav format)
alert_sound_path = "alert.wav"

# Initialize car status and the set for wrong-way cars
car_status = {}
wrong_way_cars = set()

# Initialize a DataFrame to store wrong-way car info
wrong_way_data = pd.DataFrame(columns=["Car ID", "Timestamp"])

# Initialize wrong-way counts for each area
wrong_way_counts = {'Area 1': 0, 'Area 2': 0}

# Define minimum and maximum bounding box size (width, height)
MIN_SIZE = (30, 30)  # Minimum width and height of bounding boxes
MAX_SIZE = (300, 300)  # Maximum width and height of bounding boxes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.track(frame, persist=True)

    car_count = 0  # Count of cars
    truck_count = 0  # Count of trucks

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        ids = result.boxes.id.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            obj_class = int(classes[i])
            obj_id = int(ids[i])

            width = x2 - x1
            height = y2 - y1

            if class_list[obj_class] == "car" and MIN_SIZE[0] < width < MAX_SIZE[0] and MIN_SIZE[1] < height < MAX_SIZE[1]:
                car_count += 1  # Increment car count
                cx = (x1 + x2) // 2
                cy = y2

                in_area1 = cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False) >= 0
                in_area2 = cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False) >= 0

                if obj_id not in car_status:
                    car_status[obj_id] = {"in_area1": False, "in_area2": False, "wrong_way": False, "saved": False}

                if in_area1:
                    car_status[obj_id]["in_area1"] = True
                if in_area2:
                    car_status[obj_id]["in_area2"] = True

                # Check for wrong-way detection
                if car_status[obj_id]["in_area1"] and in_area2 and not car_status[obj_id]["wrong_way"]:
                    car_status[obj_id]["wrong_way"] = True
                    wrong_way_counts['Area 1'] += 1  # Increment Area 1 wrong-way count

                    # Get the current timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Play alert sound and send email alert in separate threads
                    threading.Thread(target=play_alert_sound).start()

                    # Add the image path to the email alert
                    timestamp_for_image = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    car_image_path = os.path.join(save_dir, f'car_{obj_id}_{timestamp_for_image}.png')

                    # Ensure bounding box is within the frame bounds
                    if 0 <= y1 < frame.shape[0] and 0 <= y2 < frame.shape[0] and 0 <= x1 < frame.shape[1] and 0 <= x2 < frame.shape[1]:
                        car_image = frame[y1:y2, x1:x2]
                        saved = cv2.imwrite(car_image_path, car_image)
                        if saved:
                            print(f"Image saved for car ID {obj_id} at {car_image_path}")
                        else:
                            print(f"Failed to save image for car ID {obj_id}")

                        car_status[obj_id]["saved"] = True
                        wrong_way_cars.add(obj_id)

                        # Send email alert with the image attachment
                        threading.Thread(target=send_email_alert, args=(obj_id, timestamp, car_image_path)).start()

                        # Add a new row with the car ID and timestamp to the DataFrame
                        new_row = pd.DataFrame({"Car ID": [obj_id], "Timestamp": [timestamp_for_image]})
                        wrong_way_data = pd.concat([wrong_way_data, new_row], ignore_index=True)

                if car_status[obj_id]["in_area2"] and in_area1 and not car_status[obj_id]["wrong_way"]:
                    car_status[obj_id]["wrong_way"] = True
                    wrong_way_counts['Area 2'] += 1  # Increment Area 2 wrong-way count

                    # Get the current timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Play alert sound and send email alert in separate threads
                    threading.Thread(target=play_alert_sound).start()

                    # Add the image path to the email alert
                    timestamp_for_image = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    car_image_path = os.path.join(save_dir, f'car_{obj_id}_{timestamp_for_image}.png')

                    # Ensure bounding box is within the frame bounds
                    if 0 <= y1 < frame.shape[0] and 0 <= y2 < frame.shape[0] and 0 <= x1 < frame.shape[1] and 0 <= x2 < frame.shape[1]:
                        car_image = frame[y1:y2, x1:x2]
                        saved = cv2.imwrite(car_image_path, car_image)
                        if saved:
                            print(f"Image saved for car ID {obj_id} at {car_image_path}")
                        else:
                            print(f"Failed to save image for car ID {obj_id}")

                        car_status[obj_id]["saved"] = True
                        wrong_way_cars.add(obj_id)

                        # Send email alert with the image attachment
                        threading.Thread(target=send_email_alert, args=(obj_id, timestamp, car_image_path)).start()

                        # Add a new row with the car ID and timestamp to the DataFrame
                        new_row = pd.DataFrame({"Car ID": [obj_id], "Timestamp": [timestamp_for_image]})
                        wrong_way_data = pd.concat([wrong_way_data, new_row], ignore_index=True)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cvzone.putTextRect(frame, f'ID: {obj_id}', (x1, y1 - 5), 1, 1)

                if car_status[obj_id]["wrong_way"]:
                    cvzone.putTextRect(frame, f'Wrong Way {obj_id}', (x1, y2 - 20), 1, 1, colorR=(0, 0, 255))

            elif class_list[obj_class] == "truck":
                truck_count += 1  # Increment truck count and can also add similar size checks if needed

    # Draw area outlines
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 255, 255), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 255, 255), 2)

    # Display the counts of cars and trucks at the top left corner
    cvzone.putTextRect(frame, f'Cars: {car_count}', (10, 30), 1, 2, colorR=(0, 255, 0))
    cvzone.putTextRect(frame, f'Trucks: {truck_count}', (10, 60), 1, 2, colorR=(255, 0, 0))

    cvzone.putTextRect(frame, f'Wrong way Cars: {len(wrong_way_cars)}', (10, 90), 1, 2, colorR=(0, 255, 0))
    cv2.imshow("output", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Save the DataFrame to an Excel file after processing is complete
wrong_way_data.to_excel("wrong_way_cars.xlsx", index=False)

# Plot the bar graph for wrong-way counts
plt.figure(figsize=(8, 5))
plt.bar(wrong_way_counts.keys(), wrong_way_counts.values(), color='red', alpha=0.7)
plt.title('Count of Wrong-Way Vehicle Detections by Area')
plt.xlabel('Area')
plt.ylabel('Number of Wrong-Way Detections')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

cap.release()
cv2.destroyAllWindows()
