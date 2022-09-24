import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from collections import Counter


def crop_image_to_bounding_box(image_path, weight_path, cfg_path):
    threshold = 0.5
    net = cv2.dnn.readNet(weight_path, cfg_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    img = cv2.imread(image_path)
    img = cv2.resize(img, (1000, 700))
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                h = int(detection[2] * width)
                w = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(img, (x, y), (x + h, y + w), (255, 250, 250), 2)
            img = img[y:y + h, x:x + h]
    height, width = img.shape[:2]
    resized_image = cv2.resize(img, (3 * width, 3 * height), interpolation=cv2.INTER_CUBIC)
    return resized_image


def get_histogram_counter(image):
    embedded_vector = np.array(image).flatten()
    input_histogram_count = Counter(embedded_vector)
    input_histogram = []
    for i in range(256):
        if i in input_histogram_count.keys():
            input_histogram.append(input_histogram_count[i])
        else:
            input_histogram.append(0)
    return input_histogram


def get_list_of_selected_images(path):
    selected_images = [f for f in listdir(path) if isfile(join(path, f))]
    return selected_images


def get_euclidean_distance(histogram_1, histogram_2):
    distances = 0
    for i in range(len(histogram_1)):
        # distances = distance.euclidean(histogram_1[i], histogram_2[i])
        distances = np.linalg.norm(histogram_1[i] - histogram_2[i])
    return distances


def get_euclidean_distance_list(selected_image_list, selected_dataset_path, input_image, input_histogram, euclidean_distances, weight_path, cfg_path):
    for image in selected_image_list:
        if image == input_image:
            continue
        else:
            try:
                get_crop_image = crop_image_to_bounding_box(selected_dataset_path + "\\" + image, weight_path, cfg_path)
            except Exception as e:
                # print(f"failed to get cropped image for {image}")
                selected_image_list.remove(image)
                continue
            get_histogram = get_histogram_counter(get_crop_image)
            get_euclidean = get_euclidean_distance(input_histogram, get_histogram)
            euclidean_distances[image] = get_euclidean
    return euclidean_distances




