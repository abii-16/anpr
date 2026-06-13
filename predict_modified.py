import hydra
import torch
import os
import cv2
import easyocr
import re
import Levenshtein
from collections import defaultdict
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

add_safe_globals([DetectionModel])

reader = easyocr.Reader(['en'], gpu=True)
unique_plate_results = {}  # {plate: timestamp}


def slow_down_video_by_duplication(input_video_path, output_video_path, slow_factor=2):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {input_video_path}")
        return input_video_path

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for _ in range(slow_factor):
            out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Slowed (x{slow_factor}) video saved to: {output_video_path}")
    return output_video_path


def perform_ocr_on_image(img, coordinates):
    x1, y1, x2, y2 = map(int, coordinates)
    cropped_img = img[y1:y2, x1:x2]
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray_img)
    text = ""
    for res in results:
        if len(res[1].replace(" ", "")) >= 7 and res[2] > 0.3:
            text = res[1].replace(" ", "").upper()
    return text


def is_valid_plate(plate):
    # Regex pattern for Indian vehicle number plates
    # pattern = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$"
    pattern = r"^[A-Za-z0-9]{4}\s?[A-Za-z]{3}$"
    return bool(re.fullmatch(pattern, plate))



def group_similar_plates(plates_dict, threshold=2):
    groups = []
    used = set()
    plates = list(plates_dict.items())
    for i, (plate1, time1) in enumerate(plates):
        if plate1 in used:
            continue
        group = [(plate1, time1)]
        used.add(plate1)
        for j in range(i + 1, len(plates)):
            plate2, time2 = plates[j]
            if plate2 not in used and Levenshtein.distance(plate1, plate2) <= threshold:
                group.append((plate2, time2))
                used.add(plate2)
        groups.append(group)
    return groups


class DetectionPredictor(BasePredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.output_writer = None

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou,
                                        agnostic=self.args.agnostic_nms, max_det=self.args.max_det)
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        if len(im.shape) == 3:
            im = im[None]
        im0 = im0.copy()
        frame = getattr(self.dataset, 'frame', 0)
        timestamp = frame / 30.0

        self.data_path = p
        self.annotator = self.get_annotator(im0)
        self.seen += 1
        det = preds[idx]
        self.all_outputs.append(det)

        # Init writer
        if self.output_writer is None:
            h, w = im0.shape[:2]
            out_path = self.save_dir / "output_with_plates.mp4"
            self.output_writer = cv2.VideoWriter(str(out_path),
                                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                                 30, (w, h))
            print(f"🎥 Output will be saved to: {out_path}")

        for *xyxy, conf, cls in reversed(det):
            plate = perform_ocr_on_image(im0, xyxy)
            plate_clean = re.sub(r'\s+', '', plate.upper())
            if is_valid_plate(plate_clean):
                unique_plate_results[plate_clean] = timestamp

            if self.args.save or self.args.save_crop or self.args.show:
                c = int(cls)
                self.annotator.box_label(xyxy, plate_clean, color=colors(c, True))
                if self.args.save_crop:
                    imc = im0.copy()
                    save_one_box(xyxy, imc,
                                 file=self.save_dir / 'crops' / f'{self.data_path.stem}.jpg',
                                 BGR=True)

        result_frame = self.annotator.result()
        self.output_writer.write(result_frame)
        return f"{p.name} done\n"

    def __del__(self):
        if self.output_writer:
            self.output_writer.release()
            print("✅ Output video writer released")


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"

    if str(cfg.source).endswith('.mp4'):
        input_video = str(cfg.source)
        slowed_video = input_video.replace('.mp4', '_slowdup.mp4')
        cfg.source = slow_down_video_by_duplication(input_video, slowed_video, slow_factor=4)

    predictor = DetectionPredictor(cfg)
    predictor()

    grouped = group_similar_plates(unique_plate_results)
    save_dir = predictor.save_dir
    output_txt_path = os.path.join(save_dir, "final_detected_plates.txt")

    with open(output_txt_path, "w") as f:
        for group in grouped:
            last_plate, last_time = group[-1]
            timestamp_str = f"{int(last_time // 60):02d}:{int(last_time % 60):02d}"
            output = f"{last_plate} at {timestamp_str} (mm:ss)"
            print(output)
            f.write(output + "\n")

    print(f"\n📝 Saved final plate results to: {output_txt_path}")

    if os.path.exists(cfg.source) and cfg.source.endswith('_slowdup.mp4'):
        os.remove(cfg.source)
        print(f"🧹 Removed temporary slowed video: {cfg.source}")


if __name__ == "__main__":
    predict()




# # Ultralytics YOLO 🚀, GPL-3.0 license 
# #keeping fps same use this
# import hydra
# import torch
# import os
# import cv2
# import easyocr

# from torch.serialization import add_safe_globals
# from ultralytics.nn.tasks import DetectionModel
# add_safe_globals([DetectionModel])  # Needed to deserialize YOLOv8 weights safely

# from ultralytics.yolo.engine.predictor import BasePredictor
# from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
# from ultralytics.yolo.utils.checks import check_imgsz
# from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

# # ------------------------ Slow Down Video Function ------------------------
# def slow_down_video_by_duplication(input_video_path, output_video_path, slow_factor=2):
#     cap = cv2.VideoCapture(input_video_path)
#     if not cap.isOpened():
#         print(f" Error: Cannot open {input_video_path}")
#         return input_video_path  # Fallback to original

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         for _ in range(slow_factor):
#             out.write(frame)

#     cap.release()
#     out.release()
#     print(f"Slowed (x{slow_factor}) video saved to: {output_video_path}")
#     return output_video_path

# # ------------------------ OCR ------------------------
# reader = easyocr.Reader(['en'], gpu=True)

# def perform_ocr_on_image(img, coordinates):
#     x, y, w, h = map(int, coordinates)
#     cropped_img = img[y:h, x:w]
#     gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
#     results = reader.readtext(gray_img)

#     text = ""
#     for res in results:
#         if len(results) == 1 or (len(res[1]) > 6 and res[2] > 0.2):
#             text = res[1]
#     return str(text)

# # ------------------------ Custom Predictor ------------------------
# class DetectionPredictor(BasePredictor):

#     def get_annotator(self, img):
#         return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

#     def preprocess(self, img):
#         img = torch.from_numpy(img).to(self.model.device)
#         img = img.half() if self.model.fp16 else img.float()
#         img /= 255
#         return img

#     def postprocess(self, preds, img, orig_img):
#         preds = ops.non_max_suppression(preds,
#                                         self.args.conf,
#                                         self.args.iou,
#                                         agnostic=self.args.agnostic_nms,
#                                         max_det=self.args.max_det)

#         for i, pred in enumerate(preds):
#             shape = orig_img[i].shape if self.webcam else orig_img.shape
#             pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
#         return preds

#     def write_results(self, idx, preds, batch):
#         p, im, im0 = batch
#         log_string = ""
#         if len(im.shape) == 3:
#             im = im[None]
#         self.seen += 1
#         im0 = im0.copy()
#         frame = getattr(self.dataset, 'frame', 0)

#         self.data_path = p
#         self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
#         log_string += '%gx%g ' % im.shape[2:]
#         self.annotator = self.get_annotator(im0)

#         det = preds[idx]
#         self.all_outputs.append(det)
#         if len(det) == 0:
#             return log_string

#         for c in det[:, 5].unique():
#             n = (det[:, 5] == c).sum()
#             log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

#         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
#         for *xyxy, conf, cls in reversed(det):
#             if self.args.save_txt:
#                 xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
#                 line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)
#                 with open(f'{self.txt_path}.txt', 'a') as f:
#                     f.write(('%g ' * len(line)).rstrip() % line + '\n')

#             if self.args.save or self.args.save_crop or self.args.show:
#                 c = int(cls)
#                 label = None if self.args.hide_labels else (
#                     self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                
#                 text_ocr = perform_ocr_on_image(im0, xyxy)
#                 label = text_ocr
#                 self.annotator.box_label(xyxy, label, color=colors(c, True))

#             if self.args.save_crop:
#                 imc = im0.copy()
#                 save_one_box(xyxy,
#                              imc,
#                              file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
#                              BGR=True)

#         return log_string

# # ------------------------ Main Entry Point ------------------------
# @hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
# def predict(cfg):
#     cfg.model = cfg.model or "yolov8n.pt"
#     cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
#     cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"

#     # Slow down video (frame duplication)
#     if str(cfg.source).endswith('.mp4'):
#         input_video = str(cfg.source)
#         slowed_video = input_video.replace('.mp4', '_slowdup.mp4')
#         cfg.source = slow_down_video_by_duplication(input_video, slowed_video, slow_factor=4)

#     predictor = DetectionPredictor(cfg)
#     predictor()

#     # Optional cleanup
#     if os.path.exists(cfg.source) and cfg.source.endswith('_slowdup.mp4'):
#         os.remove(cfg.source)
#         print(f"🧹 Removed temporary slowed video: {cfg.source}")

# if __name__ == "__main__":
#     predict()



# # Ultralytics YOLO 🚀, GPL-3.0 license
# #2x slower
# import hydra
# import torch
# import cv2
# import os

# from ultralytics.yolo.engine.predictor import BasePredictor
# from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
# from ultralytics.yolo.utils.checks import check_imgsz
# from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

# import easyocr
# from torch.serialization import add_safe_globals
# from ultralytics.nn.tasks import DetectionModel  # this is the critical class

# # Add YOLOv8 class to safe globals so the weights can be deserialized
# add_safe_globals([DetectionModel])

# reader = easyocr.Reader(['en'], gpu=True)

# # ✅ Step 1: Slow down the input video
# def slow_down_video(input_video_path, output_video_path, slow_factor=0.5):
#     cap = cv2.VideoCapture(input_video_path)
#     if not cap.isOpened():
#         print(f"❌ Error: Cannot open {input_video_path}")
#         return input_video_path  # fallback

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     slowed_fps = fps * slow_factor
#     out = cv2.VideoWriter(output_video_path, fourcc, slowed_fps, (width, height))

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         out.write(frame)

#     cap.release()
#     out.release()
#     print(f"✅ Slowed video saved to: {output_video_path}")
#     return output_video_path


# def perform_ocr_on_image(img, coordinates):
#     x, y, w, h = map(int, coordinates)
#     cropped_img = img[y:h, x:w]

#     gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
#     results = reader.readtext(gray_img)

#     text = ""
#     for res in results:
#         if len(results) == 1 or (len(res[1]) > 6 and res[2] > 0.2):
#             text = res[1]

#     return str(text)


# class DetectionPredictor(BasePredictor):

#     def get_annotator(self, img):
#         return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

#     def preprocess(self, img):
#         img = torch.from_numpy(img).to(self.model.device)
#         img = img.half() if self.model.fp16 else img.float()
#         img /= 255
#         return img

#     def postprocess(self, preds, img, orig_img):
#         preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou,
#                                         agnostic=self.args.agnostic_nms, max_det=self.args.max_det)
#         for i, pred in enumerate(preds):
#             shape = orig_img[i].shape if self.webcam else orig_img.shape
#             pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
#         return preds

#     def write_results(self, idx, preds, batch):
#         p, im, im0 = batch
#         log_string = ""
#         if len(im.shape) == 3:
#             im = im[None]
#         self.seen += 1
#         im0 = im0.copy()

#         frame = self.dataset.count if self.webcam else getattr(self.dataset, 'frame', 0)
#         self.data_path = p
#         self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
#         log_string += '%gx%g ' % im.shape[2:]
#         self.annotator = self.get_annotator(im0)

#         det = preds[idx]
#         self.all_outputs.append(det)
#         if len(det) == 0:
#             return log_string
#         for c in det[:, 5].unique():
#             n = (det[:, 5] == c).sum()
#             log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
#         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

#         for *xyxy, conf, cls in reversed(det):
#             if self.args.save_txt:
#                 xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
#                 line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)
#                 with open(f'{self.txt_path}.txt', 'a') as f:
#                     f.write(('%g ' * len(line)).rstrip() % line + '\n')

#             if self.args.save or self.args.save_crop or self.args.show:
#                 c = int(cls)
#                 label = None if self.args.hide_labels else (
#                     self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                
#                 text_ocr = perform_ocr_on_image(im0, xyxy)
#                 label = text_ocr
#                 self.annotator.box_label(xyxy, label, color=colors(c, True))

#             if self.args.save_crop:
#                 imc = im0.copy()
#                 save_one_box(xyxy, imc,
#                              file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
#                              BGR=True)

#         return log_string


# @hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
# def predict(cfg):
#     # ✅ Step 2: Slow down if MP4
#     if cfg.source.endswith('.mp4'):
#         slowed_video_path = 'vid2_slowed.mp4'
#         cfg.source = slow_down_video(cfg.source, slowed_video_path, slow_factor=0.5)

#     cfg.model = cfg.model or "yolov8n.pt"
#     cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
#     cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"

#     predictor = DetectionPredictor(cfg)
#     predictor()


# if __name__ == "__main__":
#     predict()



# # Ultralytics YOLO 🚀, GPL-3.0 license
#original without slowing down
# import hydra
# import torch

# from ultralytics.yolo.engine.predictor import BasePredictor
# from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
# from ultralytics.yolo.utils.checks import check_imgsz
# from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


# import easyocr
# import cv2
# from torch.serialization import add_safe_globals
# from ultralytics.nn.tasks import DetectionModel  # this is the critical class

# # Add YOLOv8 class to safe globals so the weights can be deserialized
# add_safe_globals([DetectionModel])

# reader = easyocr.Reader(['en'], gpu=True)

# def perform_ocr_on_image(img, coordinates):
#     x, y, w, h = map(int, coordinates)
#     cropped_img = img[y:h, x:w]

#     gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
#     results = reader.readtext(gray_img)

#     text = ""
#     for res in results:
#         if len(results) == 1 or (len(res[1]) > 6 and res[2] > 0.2):
#             text = res[1]

#     return str(text)


# class DetectionPredictor(BasePredictor):

#     def get_annotator(self, img):
#         return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

#     def preprocess(self, img):
#         img = torch.from_numpy(img).to(self.model.device)
#         img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
#         img /= 255  # 0 - 255 to 0.0 - 1.0
#         return img

#     def postprocess(self, preds, img, orig_img):
#         preds = ops.non_max_suppression(preds,
#                                         self.args.conf,
#                                         self.args.iou,
#                                         agnostic=self.args.agnostic_nms,
#                                         max_det=self.args.max_det)

#         for i, pred in enumerate(preds):
#             shape = orig_img[i].shape if self.webcam else orig_img.shape
#             pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

#         return preds

#     def write_results(self, idx, preds, batch):
#         p, im, im0 = batch
#         log_string = ""
#         if len(im.shape) == 3:
#             im = im[None]  # expand for batch dim
#         self.seen += 1
#         im0 = im0.copy()
#         if self.webcam:  # batch_size >= 1
#             log_string += f'{idx}: '
#             frame = self.dataset.count
#         else:
#             frame = getattr(self.dataset, 'frame', 0)

#         self.data_path = p
#         # save_path = str(self.save_dir / p.name)  # im.jpg
#         self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
#         log_string += '%gx%g ' % im.shape[2:]  # print string
#         self.annotator = self.get_annotator(im0)

#         det = preds[idx]
#         self.all_outputs.append(det)
#         if len(det) == 0:
#             return log_string
#         for c in det[:, 5].unique():
#             n = (det[:, 5] == c).sum()  # detections per class
#             log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
#         # write
#         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#         for *xyxy, conf, cls in reversed(det):
#             if self.args.save_txt:  # Write to file
#                 xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                 line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
#                 with open(f'{self.txt_path}.txt', 'a') as f:
#                     f.write(('%g ' * len(line)).rstrip() % line + '\n')

#             if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
#                 c = int(cls)  # integer class
#                 label = None if self.args.hide_labels else (
#                     self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                
                
#                 text_ocr = perform_ocr_on_image(im0,xyxy)
#                 label = text_ocr 
                
#                 self.annotator.box_label(xyxy, label, color=colors(c, True))
#             if self.args.save_crop:
#                 imc = im0.copy()
#                 save_one_box(xyxy,
#                              imc,
#                              file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
#                              BGR=True)

#         return log_string


# @hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
# def predict(cfg):
#     cfg.model = cfg.model or "yolov8n.pt" #"best.pt"  
#     cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
#     cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
#     predictor = DetectionPredictor(cfg)
#     predictor()


# if __name__ == "__main__":
#     predict()
