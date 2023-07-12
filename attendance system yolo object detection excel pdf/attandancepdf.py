import cv2
import datetime
import os
import csv
import numpy as np
from PyPDF2 import PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet

# Load YOLOv3 model
net = cv2.dnn.readNet("face-yolov3-tiny_41000.weights", "face-yolov3-tiny.cfg")

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to get timestamp with date and time
def get_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime("%B %d, %Y %I:%M %p")
    return timestamp

# Function to apply Non-Maximum Suppression (NMS) to bounding boxes
def apply_nms(boxes, scores, threshold=0.5):
    indices = cv2.dnn.NMSBoxes(boxes, scores, threshold, threshold - 0.1)
    return indices.flatten() if len(indices) > 0 else []

# Function to crop and annotate frame
def crop_and_annotate_frame(frame, frame_count):
    height, width, _ = frame.shape

    # Detect objects using YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # Get class labels, confidences, and bounding boxes
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply Non-Maximum Suppression (NMS) to bounding boxes
    indices = apply_nms(boxes, confidences)

    # Crop and annotate frame with timestamp
    cropped_images = []
    for i in indices:
        x, y, w, h = boxes[i]
        crop = frame[y:y + h, x:x + w]

        # Check if the cropped frame has a valid size
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            # Add timestamp on the right side of the cropped frame
            timestamp = get_timestamp()
            cv2.putText(crop, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Save the cropped and annotated frame
            output_path = f"output/frame_{frame_count}.jpg"
            cv2.imwrite(output_path, crop)

            # Append the cropped image to the list
            cropped_images.append(output_path)

            # Increment frame count
            frame_count += 1

    return cropped_images, frame_count

# Create PDF with cropped frames and table
def create_pdf_with_frames(cropped_images, csv_writer, output_folder):
    doc = SimpleDocTemplate(f"{output_folder}/frames_with_timestamp.pdf", pagesize=letter)
    elements = []

    # Add table with heading
    table_data = [["BLUE BRAIN ROBOTICS", "Madurai, Tamil Nadu"]]
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.gray),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
    ]))
    elements.append(table)

    # Add cropped frames and timestamps to the PDF
    styles = getSampleStyleSheet()
    for image_path in cropped_images:
        img = Image(image_path, width=400, height=300)
        elements.append(img)

        # Add timestamp text on the right side
        timestamp_text = get_timestamp()
        p = Paragraph(timestamp_text, styles["BodyText"])
        elements.append(p)

        # Write timestamp to CSV file
        csv_writer.writerow([timestamp_text])

    # Build PDF
    doc.build(elements)

# Main program
if __name__ == '__main__':
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Open webcam
    cap = cv2.VideoCapture(0)  # Change the index if you have multiple webcams

    # Get webcam frame dimensions
    _, frame = cap.read()
    frame_height, frame_width, _ = frame.shape

    # Create window to display the frame
    cv2.namedWindow("Live Stream", cv2.WINDOW_NORMAL)

    # Variables for tracking time intervals
    interval = 12  # Time interval in seconds
    frame_count = 0
    cropped_images = []
    output_folder = None
    csv_file = None
    csv_writer = None
    start_time = datetime.datetime.now()

    while True:
        # Capture frame from webcam
        ret, frame = cap.read()

        # Skip if frame not captured
        if not ret:
            break

        # Get current time
        current_time = datetime.datetime.now()

        # Check if it's time to create a new PDF and folder
        if output_folder is None or (current_time - start_time).total_seconds() >= interval:
            # Create output folder with a unique name using current timestamp
            output_folder = f"output/{current_time.strftime('%Y%m%d%H%M%S')}"
            os.makedirs(output_folder, exist_ok=True)

            # Create CSV file for logging timestamps
            csv_file = os.path.join(output_folder, "timestamps.csv")
            csv_writer = csv.writer(open(csv_file, "w", newline=""))
            csv_writer.writerow(["Timestamp"])

            # Reset variables for the next interval
            frame_count = 0

        # Crop and annotate frame
        cropped, frame_count = crop_and_annotate_frame(frame, frame_count)
        cropped_images.extend(cropped)

        # Check if it's time to create the PDF
        if (current_time - start_time).total_seconds() >= interval:
            # Create PDF with cropped frames and table
            create_pdf_with_frames(cropped_images, csv_writer, output_folder)

            # Reset variables for the next interval
            cropped_images = []
            start_time = current_time

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

