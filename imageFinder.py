import image2vect
import os, shutil
from collections import defaultdict
import random
import matplotlib.pyplot as plt

euclidean_distances = {}
precisions = []
recalls = []
cur_dir = os.getcwd()
selected_dataset_path = cur_dir + "\selected"
weight_path = cur_dir + "\.yoloface\yolov3-tiny_face.weights"
cfg_path = cur_dir + "\.yoloface\yolov3_tiny_face.cfg"
output_path = cur_dir + "\outputs"
celebrity_num = 29.0
tau = [8500.0, 9500.0, 10000.0, 11500.0, 13000.0, 14500.0, 15000.0, 16000.0, 16500.0, 17000.0]
if not os.path.exists(selected_dataset_path):
    os.mkdir(selected_dataset_path)
if not os.path.exists(output_path):
    os.mkdir(output_path)


def make_dataset():
    with open(cur_dir + "\selected_ids.txt", 'r') as f:
        selected_ids = f.read()

    selected_ids = selected_ids.split("\n")
    for id in selected_ids:
        if len(id) == 0:
            selected_ids.remove(id)

    celeb_dict = {}
    for line in open(cur_dir + "\identity_CelebA.txt", 'r'):
        split = line.strip().split(' ', 1)
        celeb_dict[split[0]] = split[1]

    new_celeb_dict = defaultdict(list)
    for keys, values in celeb_dict.items():
        new_celeb_dict[values].append(keys)

    for id in selected_ids:
        if new_celeb_dict[id]:
            for celeb_img in new_celeb_dict[id]:
                src = cur_dir + f"\img_celeba\{celeb_img}"
                dst = selected_dataset_path
                shutil.copy(src, dst)


def get_precision_curve(precisions, tau, ax):
    ax.plot(tau, precisions)
    ax.set_title('Precision-Tau Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Tau')
    plt.savefig(output_path + "\precision.png")


def get_recall_curve(recalls, tau, ax):
    ax.plot(tau, recalls)
    ax.set_title('Recall-Tau Curve')
    ax.set_ylabel('Recall')
    ax.set_xlabel('Tau')
    plt.savefig(output_path + r"\recall.png")


def get_precisions_and_recalls():
    make_dataset()
    selected_images = image2vect.get_list_of_selected_images(selected_dataset_path)
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    selected_tau_values = []
    for i in range(2):
        matched_image_list = []
        input_image = random.choice(selected_images)
        print(f"Input image is {input_image}")
        input_image_cropped = image2vect.crop_image_to_bounding_box(selected_dataset_path + "\\" + input_image,
                                                                    weight_path, cfg_path)
        input_histogram = image2vect.get_histogram_counter(input_image_cropped)
        euclidean_distance_list = image2vect.get_euclidean_distance_list(selected_images, selected_dataset_path,
                                                                         input_image, input_histogram,
                                                                         euclidean_distances, weight_path, cfg_path)
        tau_select = random.choice(tau)
        selected_tau_values.append(tau_select)
        for image in euclidean_distance_list:
            if euclidean_distance_list[image] <= tau_select:
                matched_image_list.append(image)
        precision = len(matched_image_list) / len(selected_images)
        recall = len(matched_image_list) / celebrity_num
        precisions.append(precision)
        get_precision_curve(precisions,)
        recalls.append(recall)
    return precisions, recalls, selected_tau_values


def _main():
    # precision_list, recall_list, selected_tau = get_precisions_and_recalls()
    # print(f"Precision values are - {precision_list}")
    # print(f"Recall values are - {recall_list}")
    # print(f"Selected tau values are - {selected_tau}")
    precision_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.9, 0.6]
    get_precision_curve(precision_list, tau)
    #get_recall_curve(recall_list, selected_tau)


if __name__ == '__main__':
    _main()
