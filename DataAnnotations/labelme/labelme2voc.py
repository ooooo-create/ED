import argparse
from pathlib import Path

import imgviz
import labelme
import numpy as np


def main():
    init_path = Path(__file__).parent
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", default=init_path / "annotations", help="Input annotate directory")
    parser.add_argument("--output_dir", default=init_path, help="Output dataset directory")
    parser.add_argument("--labels", default=init_path / "label.txt", help="labels file")
    args = parser.parse_args()
    args.noviz = False

    if not Path.exists(args.output_dir):
        Path(args.output_dir).mkdir()
    Path(args.output_dir / "JPEGImages").mkdir()
    Path(args.output_dir / "SegmentationClass").mkdir()
    Path(args.output_dir / "SegmentationClassPNG").mkdir()

    if not args.noviz:
        Path(args.output_dir / "SegmentationClassVisualization").mkdir()

    print("Creating dataset:", args.output_dir)

    if Path.exists(args.labels):
        with open(args.labels) as f:
            labels = [label.strip() for label in f if label]
    else:
        # labels = [label.strip() for label in args.labels.split(",")]
        print("label.txt is missing, please make a file for it")

    class_names = []
    class_names_to_id = {}
    for i, label in enumerate(labels):
        class_id = i - 1
        class_name = label.strip()
        class_names_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    # class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = Path(args.output_dir / "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Save class_names:", out_class_names_file)

    for filename in sorted([x for x in Path(args.input_dir).glob("*.json")]):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)
        base = Path(filename).stem
        out_img_file = Path(args.output_dir / "JPEGImages" / base).with_suffix(".jpg")
        out_lbl_file = Path(args.output_dir / "SegmentationClass" / base).with_suffix(".npy")
        out_png_file = Path(args.output_dir / "SegmentationClassPNG" / base).with_suffix(".png")
        if not args.noviz:
            out_viz_file: Path = Path(args.output_dir / "SegmentationClassVisualization" / base).with_suffix(".jpg")

        with open(out_img_file, "wb") as f:
            f.write(label_file.imageData)
        img = labelme.utils.img_data_to_arr(label_file.imageData)

        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_names_to_id
        )
        labelme.utils.lblsave(out_png_file, lbl)

        np.save(out_lbl_file, lbl)

        if not args.noviz:
            viz = imgviz.label2rgb(
                lbl,
                imgviz.rgb2gray(img),
                font_size=15,
                label_names=class_names,
                loc="rb",
            )
            imgviz.io.imsave(str(out_viz_file), viz)


if __name__ == "__main__":
    main()
