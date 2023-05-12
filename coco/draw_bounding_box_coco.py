import argparse
import os
from tqdm import tqdm
import cv2
from pycocotools.coco import COCO
import seaborn as sns
import random

color_maps = sns.color_palette(n_colors=100)
color_maps = [(color_map[0] * 255, color_map[1] * 255, color_map[2] * 255) for color_map in color_maps]


def get_arguments():
    parser = argparse.ArgumentParser(description="Draw coco bounding boxes")  # 2. パーサを作る

    parser.add_argument("gt", help="ground truth")
    parser.add_argument("img_dir", help="img_dir")
    parser.add_argument("out_dir", help="out_dir")
    parser.add_argument("--dt", nargs="+", help="dt")
    args = parser.parse_args()

    return args


def draw_bb_coco(gt_path, img_dir, out_dir, dt_path_list):
    cocoGt = COCO(gt_path)
    if len(dt_path_list) > 0:
        cocoDt_list = [cocoGt.loadRes(dt_path) for dt_path in dt_path_list]

    categories = cocoGt.loadCats(cocoGt.getCatIds())
    cat_id2cat_name = {cat["id"]: cat["name"] for cat in categories}

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    img_ids = cocoGt.getImgIds()
    for img_id in tqdm(img_ids):
        img = cocoGt.loadImgs(ids=img_id)[0]

        img_cv2 = cv2.imread(os.path.join(img_dir, img["file_name"]))
        img_cv2_gt = img_cv2.copy()

        Ids_gts = cocoGt.getAnnIds(imgIds=img["id"], iscrowd=None)
        gts = cocoGt.loadAnns(Ids_gts)

        img_bb_list = []
        for gt in gts:
            x_min = int(gt["bbox"][0])
            y_min = int(gt["bbox"][1])
            x_max = int(gt["bbox"][0]) + int(gt["bbox"][2])
            y_max = int(gt["bbox"][1]) + int(gt["bbox"][3])

            category_id = gt["category_id"]
            plot_one_box(
                [x_min, y_min, x_max, y_max],
                img_cv2_gt,
                cat_id2cat_name[category_id],
                color=color_maps[category_id - 1],
            )

        img_bb_list.append(img_cv2_gt)

        for cocoDt in cocoDt_list:
            Ids_dt = cocoDt.getAnnIds(imgIds=img["id"], iscrowd=None)
            dts = cocoDt.loadAnns(Ids_dt)
            img_cv2_dt = img_cv2.copy()

            for dt in dts:
                x_min = int(dt["bbox"][0])
                y_min = int(dt["bbox"][1])
                x_max = int(dt["bbox"][0]) + int(dt["bbox"][2])
                y_max = int(dt["bbox"][1]) + int(dt["bbox"][3])
                category_id = dt["category_id"]
                plot_one_box(
                    [x_min, y_min, x_max, y_max],
                    img_cv2_dt,
                    cat_id2cat_name[category_id],
                    score=dt["score"],
                    color=color_maps[category_id - 1],
                )
            img_bb_list.append(img_cv2_dt)

            img_stack = cv2.hconcat(img_bb_list)
            cv2.imwrite(os.path.join(out_dir, img["file_name"]), img_stack)


def plot_one_box(bbox, img, label, score=None, color=None):
    label = f"{label} {score:.2f}" if score is not None else label
    # Plots one bounding box on image img
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == "__main__":
    args = get_arguments()
    draw_bb_coco(args.gt, args.img_dir, args.out_dir, args.dt)
