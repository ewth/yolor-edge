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

    save_vid = opt.save_as_video

    if webcam:
        view_img = True
        save_path = str(Path(out).joinpath('webcam_output.mp4'))
        display("Using webcam as source")
    else:
        save_img = True
        save_path = str(Path(out))
        if save_vid:
            source_path = str(Path(source)).replace('/resources/sources/','').replace('/','_')
            save_path = str(Path(out).joinpath(source_path)) + '.mp4'
            display(f"Saving video to path {save_path}")
        else:
            display(f"Saving images to path {save_path}")


    # Initialise
    device = select_device(opt.device)
    display(f"Using device {opt.device}, type {device.type}")

    half = device.type != 'cpu'  # half precision only supported on CUDA

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
        dataset = LoadImages(source, img_size=imgsz, auto_size=64)

    if opt.headless:
        display("Running in headless mode")
        view_img = False

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # reserve pure green for humans
    colors = [x if not x == [0,255,0] else [[random.randint(0,125) for _ in range(3)]] for x in colors]
    colors[0] = [0,255,0]
    # might be better to use one colour regardless of class
    # colors = [[0, 255, 0]]

    classes = opt.classes

    show_details = opt.details
    stats_times = []
    stats_images = 0
    stats_detections = 0
    images_written = 0
    image_buffer = []
    image_files_buffer = []

    image_w = int(imgsz)
    image_h = int((3/4.0)*image_w)

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

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        inference_time = t2 - t1
        stats_times.append(inference_time)
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            # s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            detections = det is not None and len(det)

            human_detections = 0

            if detections:
                # Scale image if in images mode and saving to video
                if dataset.mode == 'images' and save_vid:
                    if not im0.shape[1] == image_w:
                        im0 = cv2.resize(im0, (image_w, image_h), interpolation=cv2.INTER_CUBIC)

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # print(det)
                # Print results
                detect_count = len(det)
                # @todo: calculate avg conf from detections
                # avg_conf = float(det[:, 4]).sum() / detect_count
                avg_conf = 0
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     detect_count += n
                if verbose:
                    display("%d detections in %.3fs" % (detect_count, inference_time))
                # Write results
                for *xyxy, conf, cls in det:
                    stats_detections += 1
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if cls == 0:
                        human_detections += 1
                    if show_details and (save_img or view_img):  # Add bbox to image
                        # label = "Human {conf100:.0f}%".format(name=names[int(cls)], conf100=conf*100)
                        # c1, c2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
                        # @todo: put in middle of bounding box
                        # x, y = int(c1[0] + (c2[0] - c1[0])/2), int(c1[1] + (c2[1] - c1[1])/2)
                        # cv2.putText(im0, label, (int(x), int(y)), 0, 1 / 3, (0,255,0), thickness=1, lineType=line_type)
                        if cls == 0:
                            label = "Human ({:.2f}%)".format((conf * 100))
                        else:
                            label = '%s %.2f' % (names[int(cls)], conf)
                        # plot_text_with_border((x,y), im0, label, (0,255,0))
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2, text_color=[0,0,0])
            else:
                if verbose:
                    print("No detections")

            # Print summary stats on image
            if save_img or view_img:
                # Update display based on detections
                # No detections: red box
                # Detections: green box with number of humans detected
                if human_detections < 1:
                    cv2.rectangle(im0, (0, 0), (30, 20), (0, 0, 255), -1, lineType=cv2.LINE_4)
                else:
                    cv2.rectangle(im0, (0, 0), (30, 20), (0, 255, 0), -1, lineType=cv2.LINE_4)
                    # Changing image labelling to suit project
                    # plot_text_with_border((0,20), im0, "Inf. Time: %.3fs" % inference_time)
                    cv2.putText(im0, str(human_detections).rjust(4, ' '), (0,15), 0, 0.4, (0, 0, 0), thickness=2, lineType=cv2.LINE_4)
                    # plot_text_with_border((0,40), im0, "Humans: %d" % detect_count)
                    # plot_text_with_border((0,60), im0, "Conf.: %.1f" % (avg_conf * 100))

            # Stream results
            if view_img:
                cv2.imshow('DeepRescue', im0)
                # if cv2.waitKey(1) == ord('q'):  # q to quit
                    # raise StopIteration

            # Save results (image with detections)
            # just saving to a video instead
            if save_img:
                if dataset.mode == 'images' and not save_vid:
                    cv2.imwrite(str(Path(save_path) / Path(p).name), im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        print(f"Setting up new video writer to {vid_path}")
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        # fourcc = 'mp4v'  # output video codec
                        fourcc = 'H264'
                        if dataset.mode == 'images':
                            fps = 1
                            w = int(imgsz)
                            h = int((3/4.0)*w)
                            print(f"Setting up video for {fps} FPS at {w}x{h}")
                        else:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    images_written += 1
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
    parser.add_argument('--details', action='store_true', help='Add extra details to output (boxes, labels)')
    parser.add_argument('--headless', action='store_true', help='Running in headless mode')
    parser.add_argument('--append-date', action='store_true', help='Append date time string to output path')
    parser.add_argument('--save-as-video', action='store_true', help='Save images as video (if in images mode)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--nth-frame', type=int, default=4, help='Nth frame to capture from webcam source')
    opt = parser.parse_args()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
