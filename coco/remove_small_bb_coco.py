import argparse
import collections as cl
import json
import numpy as np
from pycocotools.coco import COCO


def main():
    args = get_args()
    remove_small_bb(args.coco_input_file, args.coco_output_file, args.thresh)


def remove_small_bb(input_coco_file_name: str, output_coco_file_name: str, thresh: float):
    coco_data = COCO(input_coco_file_name)
    coco_out = coco_data.dataset
    coco_out_annotations = []

    for coco_image_info_dict in coco_out["images"]:
        id_img = coco_image_info_dict["id"]
        width_img = coco_image_info_dict["width"]
        height_img = coco_image_info_dict["height"]
        diagonal_img = np.sqrt(width_img**2 + height_img**2)
        id_annos = coco_data.getAnnIds(imgIds=id_img, iscrowd=None)
        annos = coco_data.loadAnns(id_annos)
        for anno in annos:
            width_bb = anno["bbox"][2]
            height_bb = anno["bbox"][3]
            diagonal_bb = np.sqrt(width_bb**2 + height_bb**2)

            if diagonal_bb/diagonal_img >= thresh:
                coco_out_annotations.append(anno)

    coco_out["annotations"] = coco_out_annotations
    fw = open(output_coco_file_name, "w")
    json.dump(coco_out, fw, indent=2)


def get_args():
    parser = argparse.ArgumentParser(description="Remove small bounding boxes in coco json file.")
    parser.add_argument("coco_input_file", help="Path of coco file")
    parser.add_argument("coco_output_file", help="Path of output coco file, e.g., instances.json")
    parser.add_argument("thresh", type = float, help="Thresh of small diagonal of bounding box")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
