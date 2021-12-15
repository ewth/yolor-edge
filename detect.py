import argparse
from math import ceil
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
    colors = [[0, 255, 0], [41, 126, 255], [0, 0, 229], [255, 102, 0], [24, 24, 153], [13, 105, 166], [255, 169, 41], [17, 57, 217], [77, 0, 191], [35, 217, 217], [242, 0, 145], [217, 35, 180], [217, 71, 35], [166, 82, 27], [97, 242, 0], [41, 169, 255], [204, 255, 0], [50, 15, 191], [191, 86, 15], [242, 0, 97], [0, 0, 217], [121, 15, 191], [15, 15, 191], [179, 29, 149], [255, 84, 41], [255, 255, 0], [166, 110, 27], [191, 191, 31], [0, 255, 255], [145, 242, 0], [166, 27, 166], [204, 82, 0], [242, 0, 242], [0, 153, 255], [122, 153, 0], [255, 126, 41], [51, 0, 255], [41, 255, 84], [0, 255, 204], [59, 29, 178], [0, 48, 242], [166, 27, 54], [31, 191, 191], [29, 29, 178], [177, 17, 217], [0, 191, 38], [18, 230, 187], [153, 153, 0], [177, 217, 17], [33, 0, 166], [31, 191, 159], [166, 44, 13], [41, 84, 255], [97, 17, 217], [0, 194, 242], [0, 102, 255], [31, 191, 127], [149, 29, 179], [0, 217, 43], [230, 103, 18], [99, 0, 166], [217, 35, 144], [31, 159, 191], [137, 217, 17], [255, 153, 0], [105, 166, 13], [41, 0, 204], [54, 166, 27], [31, 127, 191], [217, 130, 0], [153, 122, 0], [63, 191, 31], [0, 31, 153], [212, 41, 255], [166, 66, 0], [255, 204, 0], [191, 15, 86], [13, 166, 105], [41, 41, 255], [71, 35, 217], [24, 153, 76], [31, 191, 95], [127, 24, 153], [84, 41, 255], [61, 18, 229], [13, 166, 44], [40, 12, 153], [0, 36, 178], [138, 0, 229], [242, 48, 0], [126, 41, 255], [0, 0, 166], [0, 0, 255], [191, 15, 191], [191, 121, 15], [230, 0, 0], [191, 156, 15], [166, 0, 99], [95, 191, 31], [102, 0, 255], [217, 35, 71], [191, 15, 121], [230, 61, 18], [64, 242, 19], [24, 76, 153], [191, 50, 15], [31, 95, 191], [18, 230, 145], [0, 255, 153], [153, 50, 24], [230, 0, 230], [89, 29, 178], [0, 153, 153], [217, 144, 35], [35, 35, 217], [204, 41, 0], [153, 0, 122], [24, 50, 153], [0, 133, 166], [66, 0, 166], [242, 0, 194], [0, 130, 217], [166, 0, 33], [169, 41, 255], [217, 108, 35], [0, 66, 166], [0, 242, 242], [39, 242, 120], [166, 99, 0], [230, 37, 114], [191, 15, 15], [242, 0, 0], [159, 191, 31], [31, 63, 191], [13, 166, 135], [97, 217, 17], [0, 87, 217], [82, 166, 27]]

    # might be better to use one colour regardless of class
    # colors = [[0, 255, 0]]

    classes = opt.classes

    display_bb = opt.display_bb
    stats_times = []
    stats_images = 0
    stats_detections = 0
    

    image_w = int(imgsz)
    image_h = int((3/4.0)*image_w)

    stats_base_string = f"Model: {model_name}\nInf. Size: {imgsz} px\nConf. Thresh: {opt.conf_thres:.3f}\nIoU Thresh: {opt.iou_thres:.3f}\nDevice: {device.type}:{opt.device}"
    if classes:
        stats_base_string += " / Only: "
        for cls in classes:
            if cls <= len(names):
                stats_base_string += names[cls] + " "
            else:
                stats_base_string += f" cls{cls:d}"

    frames_counted = 0
    detect_count = 0
    avg_conf = inst_fps = run_time  = 0

    frame_stats_updated_at = 0

    video_src_width = video_src_height = 0
    running_classes = []
    running_conf = []
    running_names = []
    running_detect_count = this_frame_count = running_frame_count = 0
    prev_frame = 0
    video_resize_factor = 1
    video_resize = False
    videos_to_resize = []
    iteration_start = last_frame_check = time.time()


    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    display("Beginning inference...")
    for path, img, im0s, vid_cap in dataset:

        # @todo: sus out what img, im0s, vid_cap contain
        # want to rescale if it's too wide
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

        inst_detected_classes = []
        frames_counted += 1
        this_frame_count += 1
        running_frame_count += 1
        this_detect_count = 0

        current_frame = dataset.frame
        if current_frame == 1 or current_frame < prev_frame:
            frames_counted = 0
            display("New source, resetting stats")
            iteration_start = last_frame_check = time_synchronized()
            this_frame_count = 0
            running_classes = []
            inst_detected_classes = []
            inst_detected_conf = []
            running_conf = []
            running_names = []
            avg_conf = run_time = inst_fps = 0.

        prev_frame = current_frame
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
            inst_detected_conf = []
            inst_detected_classes = []

            if detections:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                if verbose:
                    display("%d detections in %.3fs" % (detect_count, inference_time))
                # Write results
                for *xyxy, conf, cls in det:
                    this_detect_count += 1
                    stats_detections += 1
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    if save_img or view_img:
                        cls_int = int(cls)
                        inst_detected_conf.append(float(conf))
                        inst_detected_classes.append(cls_int)
                        if display_bb:  # Add bbox to image
                            # label = "Human {conf100:.0f}%".format(name=names[int(cls)], conf100=conf*100)
                            # c1, c2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
                            # @todo: put in middle of bounding box
                            # x, y = int(c1[0] + (c2[0] - c1[0])/2), int(c1[1] + (c2[1] - c1[1])/2)
                            # cv2.putText(im0, label, (int(x), int(y)), 0, 1 / 3, (0,255,0), thickness=1, lineType=line_type)
                            #label = f"{names[cls_int].title()} {conf*100:.2f}%"
                            label = names[cls_int].title()
                            # plot_text_with_border((x,y), im0, label, (0,255,0))
                            plot_one_box(xyxy, im0, label=label, color=colors[cls_int], line_thickness=2, text_color=[0,0,0], line_type=cv2.LINE_8)
                        #if video_resize:
                                # im0 = cv2.resize(im0, (video_resize_width, video_resize_height), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            else:
                if verbose:
                    print("No detections")


            # Print summary stats on image
            if (save_img or view_img) and opt.display_info and this_frame_count > 1:
                # Display stats on image
                stat_top = 20
                # @todo: get image height for stat_bottom
                # stat_bottom = 640
                stat_incr = 20
                stat_left = 10
                # cv2.putText(im0, stats_base_string, (250,250), 0, 0.4, (250, 255, 250), thickness=2, lineType=line_type)
                running_detect_count += detect_count
                inst_detected_classes = list(set(inst_detected_classes))
                inst_detected_classes.sort()
                inst_detected_names = [names[x] for x in inst_detected_classes]


                new_running_classes = running_classes + inst_detected_classes
                new_running_classes = list(set(new_running_classes))
                if len(new_running_classes) > len(running_classes):
                    running_classes = new_running_classes
                    running_classes.sort()
                    running_names = [names[x] for x in running_classes]

                inst_avg_conf = 0
                if len(inst_detected_conf):
                    running_conf += inst_detected_conf
                    inst_avg_conf = np.average(inst_detected_conf)
                    avg_conf = np.average(running_conf)
                
                right_now = time_synchronized()
                run_time = (right_now - iteration_start)
                
                if frames_counted >= 10:
                    frame_time = time_synchronized()
                    frame_time = last_frame_check - frame_time
                    inst_fps = -1
                    if frame_time > 0:
                        last_frame_check = time_synchronized()
                        inst_fps = frames_counted / frame_time
                        frames_counted = 0


                text_scale_factor = 1 if not video_resize else 1/video_resize_factor

                plot_text_with_border(x=10,y=100, img=im0, label = 'Some unscaled text')
                plot_text_with_border(x=10,y=200, img=im0, label = 'Some unscaled Bold Text', bold_text=True)
                plot_text_with_border(x=10,y=300, img=im0, label = 'Some unscaled text but this one is really long and drags on and on and on and on and it will probably run off the screen eventually right surely there has to be an edge to this image around here somewhere')
                plot_text_with_border(x=10,y=400, img=im0, label = 'Some scaled text', scale_factor=text_scale_factor)
                plot_text_with_border(x=10,y=500, img=im0, label = 'Some MORE scaled text but this one is really long and drags on and on and on and on and it will probably run off the screen eventually right surely there has to be an edge to this image around here somewhere', scale_factor=text_scale_factor)
                plot_text_with_border(x=10,y=600, img=im0, label = 'Some Bold Text', bold_text=True, scale_factor=text_scale_factor)

                print_string = ""
                print_string += "Inst. Detections: %d\n" % detect_count
                print_string += "Inst. Classes: %s\n" % ', '.join(inst_detected_names)
                print_string += f"Inst. Avg. Conf: {inst_avg_conf*100:.2f}%\n"
                print_string += f"FPS: {inst_fps:.2f}\n"
                print_string += "Inf Time: %.3fms\n" % (1E3 * inference_time)
                print_string += f"NMS Time: %.3fms\n" % (1E3 * nms_time)

                if dataset.mode == 'video':
                    print_string += f"Frame: {dataset.frame}/{dataset.nframes}\n"
                    print_string += f"Runtime: {run_time:.2f}s\n"
                    if video_src_width > 0:
                        print_string += f"Source: {video_src_width}x{video_src_height} px"
                        if video_resize:
                            print_string += " (scaled post-process)"
                        print_string += "\n"
                print_string += "\n"
                print_string += "All Classes: %s\n" % ', '.join(running_names)
                print_string += f"Avg. Conf: {avg_conf*100:.2f}%\n"
                # print_string += f"{stats_base_string}"


                    # run_time = (datetime.now() - time_start).total_seconds()
                    # print_string += f"Run Time: %.1fs" % (run_time)
                # if fps is not None and fps >= 0.01:
                    #print_string += "\nFPS: %.2f" % fps
                # plot_text_with_border(stat_left,stat_top+(stat_incr*2), im0, print_string, scale_factor=text_scale_factor)

                # Calculate rough FPS

                # if image_mode:
                #     frames_halt = time.time() - frames_time
                #     frames_time = time.time()
                #     fps = frames_counted / frames_halt
                #     display(f"FPS: {fps:.2f}")
                # else:
                #     if frames_counted > 10:
                #         frames_halt = time.time() - frames_time
                #         frames_time = time.time()    
                #         fps = frames_counted / frames_halt
                #         display(f"FPS: {fps:.2f}")
                #         frames_counted = 0
                        

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
                        video_src_fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        video_src_width = vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        video_src_height = vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        video_src_width = int(video_src_width)
                        video_src_height = int(video_src_height)

                        if video_src_width >= 2048:
                            video_resize = True
                            video_resize_width = 1920
                            video_resize_factor = float(video_src_height) / float(video_src_width)
                            display(f"Resizing vide. Factor: {video_resize_factor:.4f}")
                            video_resize_height = video_resize_factor * video_resize_width
                            video_resize_height = int(video_resize_height)
                            videos_to_resize.append({
                                "path": save_path,
                                "w": video_resize_width,
                                "h": video_resize_height,
                                "src_w": video_src_width,
                                "src_h": video_src_height,
                                "src_fps": video_src_fps
                            })

                        else:
                            video_resize_factor = 1
                            video_resize = False
                            video_resize_width = video_src_width
                            video_resize_height = video_src_height

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), video_src_fps, (video_src_width, video_src_height))
                    vid_writer.write(im0)
                    if opt.save_frames:
                        cv2.imwrite(vid_path + f"_frame{this_frame_count}.jpg", im0)

    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()

    cv2.destroyAllWindows()
    if save_txt or save_img:
        print('Results saved to %s' % Path(out))

    # @todo: refactor into function/class/somewhere else
    # @todo: although useful, faster to copy vids and use specific software
    if False and len(videos_to_resize):
        print("Resizing videos...")
        for vid in videos_to_resize:
            src_path = Path(vid["path"])
            outfile = str(src_path)
            suffix = 'mp4'
            if len(src_path.suffix) > 0:
                outfile = outfile[:len(outfile)-len(src_path.suffix)]
                suffix = src_path.suffix
            
            outfile += '_resz' + suffix
            print(f"Resizing {src_path.name}, {vid['src_fps']}fps: {vid['src_w']}x{vid['src_h']} -> {vid['w']}x{vid['h']}")
            print(f"Writing to {outfile}")
            frame_count = 0
            vid_w = vid['w']
            vid_h = vid['h']
            vid_writer = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'mp4v'), vid['src_fps'], (vid_w, vid_h))
            cap = cv2.VideoCapture(vid['path'])
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            while True:
                if frame_count % 10 == 0:
                    print(f" frame {frame_count}/{total_frames}")
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (vid_w,vid_h), fx=0, fy=0, interpolation=cv2.INTER_AREA)
                    vid_writer.write(frame)
                    frame_count += 1
                else:
                    break


            print(f"Finished {total_frames} frames.")
            cap.release()
            vid_writer.release()
        print("Finished all videos.")
        cv2.destroyAllWindows()
                    

            

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
    parser.add_argument('--save-frames', action='store_true', help='If a video, save each frame as a JPEG')
    opt = parser.parse_args()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
