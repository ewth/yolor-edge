import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
from numpy.lib.function_base import average
import torch
import torch.backends.cudnn as cudnn

from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box, plot_text_with_border
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *
from datetime import datetime

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def display(text: str, ignore_verbose = False):
    if not ignore_verbose and not opt.verbose:
        return
    print(text)

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    verbose = opt.verbose

    if verbose:
        display("Running detection in verbose mode")

    # Setup output paths
    if opt.append_date:
        start_time = datetime.now().strftime(r"%Y%m%d%H%M%S")
        out = str(Path(out).joinpath(start_time))
        display(f"Outputting to {out}")
        
    if not os.path.exists(out):
        display(f"Creating path {out}")
        Path(out).mkdir(parents=True, exist_ok=True)  # make new output folder
    #    shutil.rmtree(out)  # delete output folder

    if webcam:
        view_img = True
        save_path = str(Path(out).joinpath('webcam_output.mp4'))
        display("Using webcam as source")
    else:
        save_img = True
        save_path = str(Path(out))
        display(f"Saving images to path {save_path}")


    # Initialise
    device = select_device(opt.device)
    display(f"Using device {opt.device}, type {device.type}")

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Get model name
    model_name = Path(weights[0]).name.replace('.pt', '')
    display(f"Using model {model_name}")

    # Load model
    model = Darknet(cfg, imgsz).cuda()
    model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
    #model = attempt_load(weights, map_location=device)  # load FP32 model
    #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    model.to(device).eval()
    if half:
        model.half()  # to FP16
        display("Using half")

    # Second-stage classifier
    # @todo: look into this
    classify = False
    if classify:
        display("Using second-stage classifier")
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, nth_frame=opt.nth_frame)
    else:
        dataset = LoadImages(source, img_size=imgsz, auto_size=64, print_output=opt.verbose)

    if opt.headless:
        display("Running in headless mode")
        view_img = False

    # Get names and colors
    names = load_classes(names)
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors = [[0, 255, 0], [141, 251, 101], [167, 98, 174], [131, 55, 102], [167, 108, 129], [119, 136, 79], [147, 189, 183], [212, 239, 196], [37, 224, 136], [241, 237, 118], [12, 135, 124], [97, 242, 214], [216, 250, 43], [222, 196, 15], [86, 35, 137], [98, 7, 192], [203, 254, 94], [235, 114, 207], [151, 6, 58], [51, 194, 99], [233, 75, 3], [166, 24, 25], [242, 167, 25], [65, 192, 158], [73, 194, 239], [54, 27, 70], [131, 245, 128], [7, 253, 199], [213, 201, 206], [230, 21, 49], [3, 156, 173], [9, 7, 192], [66, 16, 77], [214, 155, 30], [24, 138, 237], [25, 246, 191], [40, 203, 172], [108, 111, 100], [243, 106, 239], [26, 235, 93], [63, 39, 134], [189, 253, 2], [58, 244, 231], [40, 218, 117], [82, 118, 130], [238, 101, 51], [250, 236, 8], [214, 229, 55], [20, 199, 230], [142, 95, 76], [221, 240, 125], [148, 208, 96], [206, 188, 179], [175, 108, 136], [24, 10, 50], [197, 63, 55], [167, 207, 60], [126, 200, 78], [38, 124, 232], [156, 74, 44], [32, 181, 251], [172, 1, 153], [55, 133, 235], [31, 188, 71], [203, 98, 85], [129, 82, 230], [97, 121, 118], [31, 161, 29], [50, 219, 224], [251, 5, 75], [13, 221, 243], [1, 235, 140], [172, 63, 28], [58, 165, 199], [243, 91, 146], [151, 83, 219], [253, 196, 34], [19, 70, 21], [243, 45, 178], [111, 11, 65]]

    # might be better to use one colour regardless of class
    # colors = [[0, 255, 0]]

    classes = opt.classes

    display_bb = opt.display_bb
    stats_times = []
    stats_images = 0
    stats_detections = 0
    

    image_w = int(imgsz)
    image_h = int((3/4.0)*image_w)

    stats_base_string = f"Model: {model_name}\nInf. Size: {imgsz}px\nConf. Thresh: {opt.conf_thres:.3f}\nIoU Thresh: {opt.iou_thres:.3f}"
    if classes:
        stats_base_string += " / Only: "
        for cls in classes:
            if cls <= len(names):
                stats_base_string += names[cls] + " "
            else:
                stats_base_string += f" cls{cls:d}"

    frames_counted = 0
    frames_time = time.time()
    frames_rate = 0
    detect_count = avg_conf = 0

    avg_conf = 0
    detected_classes = []
    detected_conf = []
    detected_names = []
    running_detect_count = 0
    running_detect_conf = 0

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    display("Beginning inference...")
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        stats_images += 1
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        inference_time = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        nms_time = t2 - inference_time
        inference_time = inference_time - t1
        stats_times.append(inference_time)
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        detected_conf = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            # s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            detections = det is not None and len(det)

            save_path = str(Path(out) / Path(p).name)           
            detect_count = len(det)

            if detections:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                if verbose:
                    display("%d detections in %.3fs" % (detect_count, inference_time))
                # Write results
                for *xyxy, conf, cls in det:
                    detected_conf.append(float(conf))
                    stats_detections += 1
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    cls_int = int(cls)
                    if cls_int not in detected_classes:
                        detected_classes.append(cls_int)
                    if display_bb and (save_img or view_img):  # Add bbox to image
                        # label = "Human {conf100:.0f}%".format(name=names[int(cls)], conf100=conf*100)
                        # c1, c2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
                        # @todo: put in middle of bounding box
                        # x, y = int(c1[0] + (c2[0] - c1[0])/2), int(c1[1] + (c2[1] - c1[1])/2)
                        # cv2.putText(im0, label, (int(x), int(y)), 0, 1 / 3, (0,255,0), thickness=1, lineType=line_type)
                        label = f"{names[cls_int].title()} {conf*100:.2f}%"
                        # plot_text_with_border((x,y), im0, label, (0,255,0))
                        plot_one_box(xyxy, im0, label=label, color=colors[cls_int], line_thickness=2, text_color=[0,0,0])
            else:
                if verbose:
                    print("No detections")


            # Print summary stats on image
            if (save_img or view_img) and opt.display_info:
                line_type = cv2.LINE_4
                
                frames_counted += 1
                # Display stats on image
                stat_top = 20
                # @todo: get image height for stat_bottom
                # stat_bottom = 640
                stat_incr = 20
                stat_left = 10
                # cv2.putText(im0, stats_base_string, (250,250), 0, 0.4, (250, 255, 250), thickness=2, lineType=line_type)
                running_detect_count += detect_count
                detected_classes.sort()
                detected_names = [names[x] for x in detected_classes]

                plot_text_with_border(stat_left,stat_top+10, im0, 'yolor-edge: YOLOR on edge computing / Ewan Thompson 2021', border_thickness=2, font_scale=0.75, text_thickness=1)
                print_string = f"{stats_base_string}\n"
                print_string += "Inst. Detections: %d\n" % detect_count
                # print_string += "Total Detections: %d\n" % running_detect_count
                # print_string += "Frame:\n"
                print_string += "Classes: %s\n" % ', '.join(detected_names)
                print_string += "Inf Time: %.3fms\n" % (1E3 * inference_time)
                
                print_string += f"NMS Time: %.3fms\n" % (1E3 * nms_time)
                print_string += f"Run Time: %.3fs" % (t2 - t0)
                if frames_rate is not None:
                    print_string += "\nFPS: %.2f" % frames_rate

                plot_text_with_border(stat_left,stat_top+(stat_incr*2), im0, print_string)

                # plot_text_with_border((stat_left,stat_top+(stat_incr*0)), im0, 'yolor-edge: YOLOR on edge computing / Ewan Thompson 2021')
                # plot_text_with_border((stat_left,stat_top+(stat_incr*1)), im0, stats_base_string)
                # plot_text_with_border((stat_left,stat_top+(stat_incr*2)), im0, "Inst. Detections: %d" % detect_count)
                # plot_text_with_border((stat_left,stat_top+(stat_incr*3)), im0, "Total Detections: %d" % running_detect_count)
                # plot_text_with_border((stat_left,stat_top+(stat_incr*4)), im0, "Classes: %s" % ', '.join(detected_names))
                # plot_text_with_border((stat_left,stat_top+(stat_incr*4)), im0, "Avg. Conf: %s" % conf)
                # plot_text_with_border((stat_left,stat_top+(stat_incr*3)), im0, "Conf: %.1f" % (avg_conf * 100))
                # plot_text_with_border((stat_left,stat_top+(stat_incr*5)), im0, "Inf Time: %.3fms" % (1E3 * inference_time))
                # plot_text_with_border((stat_left,stat_top+(stat_incr*6)), im0, "NMS Time: %.3fms" % (1E3 * nms_time))
                # plot_text_with_border((stat_left,stat_top+(stat_incr*7)), im0, "Run Time: %.3fss" % (t0 - t2))
                # if frames_rate is not None:
                #     plot_text_with_border((stat_left,stat_top+(stat_incr*8)), im0, "FPS: %.2f" % frames_rate)
                # Calculate rough FPS
                if frames_counted > 10:
                    frames_halt = time.time() - frames_time
                    frames_time = time.time()    
                    frames_rate = frames_counted / frames_halt
                    display(f"FPS: {frames_rate:.2f}")
                    frames_counted = 0
                        

                # Update display based on detections
                # No detections: red box
                # Detections: green box with number of humans detected
                # if human_detections < 1:
                #     cv2.rectangle(im0, (0, 0), (30, 20), (0, 0, 255), -1, lineType=line_type)
                # else:
                #     cv2.rectangle(im0, (0, 0), (30, 20), (0, 255, 0), -1, lineType=line_type)
                #     # Changing image labelling to suit project
                #     # plot_text_with_border((0,20), im0, "Inf. Time: %.3fs" % inference_time)
                #     cv2.putText(im0, str(human_detections).rjust(4, ' '), (0,15), 0, 0.4, (0, 0, 0), thickness=2, lineType=line_type)
                #     # plot_text_with_border((0,40), im0, "Humans: %d" % detect_count)
                #     # plot_text_with_border((0,60), im0, "Conf.: %.1f" % (avg_conf * 100))

            # Stream results
            if view_img:
                cv2.imshow('yolor-edge', im0)
                # if cv2.waitKey(1) == ord('q'):  # q to quit
                    # raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        display(f"New vid path: {vid_path}")
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()

    cv2.destroyAllWindows()
    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print("Summary:\n\tImages: {images}\n\tDetections: {detections}\n\tAvg. Inf. Time: {avgtime:.3f}".format(images=stats_images, detections = stats_detections, avgtime = average(stats_times)))
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    parser.add_argument('--display-bb', action='store_true', help='Add bounding boxes to images for display')
    parser.add_argument('--headless', action='store_true', help='Running in headless mode')
    parser.add_argument('--append-date', action='store_true', help='Append date time string to output path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--display-info', action='store_true', help='Display info/stats on images')
    parser.add_argument('--nth-frame', type=int, default=4, help='Nth frame to capture from webcam source')
    opt = parser.parse_args()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
