import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import collections as cl
import json
import tqdm
from pycocotools.coco import COCO
import os

# data = [1, 2, 3, 4, 5, 5, 6, 6, 6, 7, 8, 9, 10]

# min_value = 1
# max_value = 20
# num_bins = 5

# sns.histplot(data, bins=num_bins, binrange=(min_value, max_value), color='skyblue', edgecolor='black')
# plt.title('Histogram')
# plt.xlabel('Values')
# plt.ylabel('Frequency')

# plt.show()


def main():
    args = get_args()
    analyze_coco(args.coco_input_file, args.out_dir, args.hist_min, args.hist_max, args.hist_bin_num)


def analyze_coco(input_coco_file_name: str, out_dir: str, hist_min: float, hist_max: float, hist_bin_num: int):
    os.makedirs(out_dir, exist_ok=True)
    coco_data = COCO(input_coco_file_name)
    categories = coco_data.loadCats(coco_data.getCatIds())
    category_id_to_name = {cat["id"]: cat["name"] for cat in categories}

    img_ids = coco_data.getImgIds()
    statistics_dict = dict.fromkeys(["images", "labels", "bboxes"])
    statistics_dict["images"] = dict.fromkeys(["total", "no_label"], 0)
    statistics_dict["images"]["total"] = len(img_ids)
    statistics_dict["labels"] = dict.fromkeys([v for v in category_id_to_name.values()] + ["total"], 0)
    statistics_dict["bboxes"] = dict.fromkeys([v for v in category_id_to_name.values()])
    for cat_name in  statistics_dict["bboxes"].keys():
        statistics_dict["bboxes"][cat_name] = {"aspect_ratio": [], "diagonal": []}

    # statistics_dict["bboxes"] = dict.fromkeys([v for v in category_id_to_name.values()], {"aspect_ratio": []})
    for img_id in tqdm.tqdm(img_ids):
        img_info = coco_data.loadImgs(ids=img_id)[0]
        id_gts = coco_data.getAnnIds(imgIds=img_id, iscrowd=None)
        gts = coco_data.loadAnns(id_gts)

        if len(gts) == 0:
            statistics_dict["images"]["no_label"] += 1

        for gt in gts:

            category_name = category_id_to_name[gt["category_id"]]

            aspect_ratio = gt["bbox"][2] / gt["bbox"][3]
            statistics_dict["bboxes"][category_name]["aspect_ratio"].append(aspect_ratio)
            statistics_dict["labels"][category_name] += 1
            statistics_dict["labels"]["total"] += 1

    # image analysis
    plt.figure(figsize=(10, 6))
    plt.bar(statistics_dict["images"].keys(), statistics_dict["images"].values())
    plt.ylabel("#images")
    plt.title("The number of images")
    plt.savefig(os.path.join(out_dir, "image_analysis.png"))

    # label analysis
    plt.figure(figsize=(10, 6))
    plt.bar(statistics_dict["labels"].keys(), statistics_dict["labels"].values())
    plt.ylabel("#bbox")
    plt.title("The number of bbox")
    plt.savefig(os.path.join(out_dir, "label_analysis.png"))

    # Bounding box analysis
    ## Aspect ratio
    for cat_name, cat_val in statistics_dict["bboxes"].items():
        plt.figure(figsize=(10, 6))
        sns.histplot(
            cat_val["aspect_ratio"],
            bins=hist_bin_num,
            binrange=(hist_min, hist_max),
            color="skyblue",
            edgecolor="black",
        )
        plt.title("Aspect ratio histgram of " + cat_name)
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(out_dir, "aspect_ratio_hist_" + cat_name + "_.png"))


def get_args():
    parser = argparse.ArgumentParser(description="Delete images that do not have labels.")
    parser.add_argument("coco_input_file", type=str, help="Path of coco file")
    parser.add_argument("out_dir", type=str, help="Path of output")
    parser.add_argument("--hist_min", default=0, type=float, help="Min value of aspect ratio histogram")
    parser.add_argument("--hist_max", default=3, type=float, help="Max value of aspect ratio histogram")
    parser.add_argument("--hist_bin_num", default=10, type=int, help="The number of bins of aspect ratio histogram")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
