import time
from pathlib import Path

import cv2
from numpy.lib.function_base import average
import torch
import torch.backends.cudnn as cudnn

from yolor.utils.datasets import LoadStreams, LoadImages
from yolor.utils.general import (non_max_suppression, apply_classifier, scale_coords, xyxy2xywh)
from yolor.utils.plots import plot_one_box, plot_text_with_border
from yolor.utils.torch_utils import select_device, load_classifier, time_synchronized

from yolor.models.models import *
from yolor.utils.datasets import *
from yolor.utils.general import *
import logging
from threading import Thread
from threading import Lock

class Detect:
    """
    Runs inference. Detects objects. The meaty parts.
    """

    # @todo: A whole lotta refactoring

    source_path: str
    output_path: str
    target_device: str
    inference_size: int
    run_name: str

    webcam_source: bool

    model_weights: str
    model_name: str
    model_config: str

    class_names_file: str
    class_names: list
    classes_restrict: list

    confidence_threshold: float
    iou_threshold: float

    nms_is_agnostic: bool

    save_text: bool
    save_images: bool
    display_images: bool
    save_video_frames: bool
    capture_nth_frame: int
    save_nth_frame: int

    display_bounding_boxes: bool
    display_bounding_box_labels: bool
    display_bounding_box_confidence: bool
    
    display_stats: bool

    display_percent_decimal = True
    mode_verbose = False
    mode_augment = False
    mode_headless = False

    stats_top_prepend = []
    stats_top_append = []
    stats_bottom_prepend = []
    stats_bottom_append = []

    system_name = "edge.nv_jx_nx"

    #
    # Juicy bits
    #
    _model = None
    _half_mode = False

    # So modest
    _have_setup = False
    _device = None
    _have_init_device = False
    _model = None
    _model_name_loaded = None
    _have_loaded_model = False
    _class_bounding_box_colours = [[0, 255, 0], [41, 126, 255], [0, 0, 229], [255, 102, 0], [24, 24, 153], [13, 105, 166], [255, 169, 41], [17, 57, 217], [77, 0, 191], [35, 217, 217], [242, 0, 145], [217, 35, 180], [217, 71, 35], [166, 82, 27], [97, 242, 0], [41, 169, 255], [204, 255, 0], [50, 15, 191], [191, 86, 15], [242, 0, 97], [0, 0, 217], [121, 15, 191], [15, 15, 191], [179, 29, 149], [255, 84, 41], [255, 255, 0], [166, 110, 27], [191, 191, 31], [0, 255, 255], [145, 242, 0], [166, 27, 166], [204, 82, 0], [242, 0, 242], [0, 153, 255], [122, 153, 0], [255, 126, 41], [51, 0, 255], [41, 255, 84], [0, 255, 204], [59, 29, 178], [0, 48, 242], [166, 27, 54], [31, 191, 191], [29, 29, 178], [177, 17, 217], [0, 191, 38], [18, 230, 187], [153, 153, 0], [177, 217, 17], [33, 0, 166], [31, 191, 159], [166, 44, 13], [41, 84, 255], [97, 17, 217], [0, 194, 242], [0, 102, 255], [31, 191, 127], [149, 29, 179], [0, 217, 43], [230, 103, 18], [99, 0, 166], [217, 35, 144], [31, 159, 191], [137, 217, 17], [255, 153, 0], [105, 166, 13], [41, 0, 204], [54, 166, 27], [31, 127, 191], [217, 130, 0], [153, 122, 0], [63, 191, 31], [0, 31, 153], [212, 41, 255], [166, 66, 0], [255, 204, 0], [191, 15, 86], [13, 166, 105], [41, 41, 255], [71, 35, 217], [24, 153, 76], [31, 191, 95], [127, 24, 153], [84, 41, 255], [61, 18, 229], [13, 166, 44], [40, 12, 153], [0, 36, 178], [138, 0, 229], [242, 48, 0], [126, 41, 255], [0, 0, 166], [0, 0, 255], [191, 15, 191], [191, 121, 15], [230, 0, 0], [191, 156, 15], [166, 0, 99], [95, 191, 31], [102, 0, 255], [217, 35, 71], [191, 15, 121], [230, 61, 18], [64, 242, 19], [24, 76, 153], [191, 50, 15], [31, 95, 191], [18, 230, 145], [0, 255, 153], [153, 50, 24], [230, 0, 230], [89, 29, 178], [0, 153, 153], [217, 144, 35], [35, 35, 217], [204, 41, 0], [153, 0, 122], [24, 50, 153], [0, 133, 166], [66, 0, 166], [242, 0, 194], [0, 130, 217], [166, 0, 33], [169, 41, 255], [217, 108, 35], [0, 66, 166], [0, 242, 242], [39, 242, 120], [166, 99, 0], [230, 37, 114], [191, 15, 15], [242, 0, 0], [159, 191, 31], [31, 63, 191], [13, 166, 135], [97, 217, 17], [0, 87, 217], [82, 166, 27]]
    _logger = None

    lock = None
    # Not reeeaalllly sure if "helping" with garbage collection is "Pythonic" so experimenting
    _help_garbage_man = False


    def __init__(
        # Lot of paramaters here...
        self,
        output_path: str,
        source_path: str,
        target_device: str,
        run_name: str,
        model_weights = "/resources/weights/yolor/yolor_p6.pt",
        model_config = "/yolor-edge/yolor/cfgyolor_p6.cfg",
        model_name = "",

        # Should be one or the other; list takes precedence
        class_names_file = "data/coco.names",
        class_names = [],
        # Empty = all classes; specificy numeric index in list to restrict (e.g. 0 = person)
        classes_restrict = [],

        inference_size = 1280,
        confidence_threshold = 0.4,
        iou_threshold = 0.5,
        nms_is_agnostic = False,

        save_text = False,
        save_images = False,
        save_video_frames = False,
        capture_nth_frame = 4,
        save_nth_frame = 20,

        display_images = False,
        display_bounding_boxes = False,
        display_bounding_box_labels = True,
        display_bounding_box_confidence = True,
        display_percent_decimal = True,
        display_stats = False,

        stats_top_append = [],
        stats_bottom_append = [],
        stats_top_prepend = [],
        stats_bottom_prepend = [],

        mode_headless = False,
        mode_augment = False,
        mode_verbose = False,

        system_name = "",

    ):
        print("[yolor.detect] Running Init")
        # Seens like a lot of double handling. Better way?

        # On 2nd thought:
        #   the one-liners look nice and all but it's hard to tell what has/hasn't been loaded.
        #   Or if things are lining up correctly with unpacking, etc.

        #
        # Paths; Device; Run name; stringy stuff
        #
        self.source_path = source_path
        self.output_path = output_path
        self.target_device = target_device
        self.run_name = run_name

        # Model details
        self.model_weights = model_weights
        self.model_config = model_config
        self.model_name = model_name

        # Object class details
        self.class_names_file = class_names_file
        self.class_names = class_names
        self.classes_restrict = classes_restrict

        # Inference configuration
        self.inference_size = inference_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.nms_is_agnostic = nms_is_agnostic

        # Saving/output details
        self.save_text = save_text
        self.save_images = save_images
        self.save_video_frames = save_video_frames
        self.display_images = display_images
        self.capture_nth_frame = capture_nth_frame
        self.save_nth_frame = save_nth_frame

        # Rendering image stuff
        self.display_stats = display_stats
        self.display_bounding_boxes = display_bounding_boxes
        self.display_bounding_box_labels = display_bounding_box_labels
        self.display_bounding_box_confidence = display_bounding_box_confidence
        self.display_percent_decimal = display_percent_decimal

        # Stuff to tack on to stats
        self.stats_top_prepend = stats_top_prepend
        self.stats_top_append = stats_top_append
        self.stats_bottom_prepend = stats_bottom_prepend
        self.stats_bottom_append = stats_bottom_append

        # Mode flags
        self.mode_verbose = mode_verbose
        self.mode_augment = mode_augment
        self.mode_headless = mode_headless

        logging.basicConfig(level=logging.DEBUG)

        self.lock = Lock()

        if system_name:
            self.system_name = system_name

        # end __init__


    def setup(self):
        """
        Setup for a detection run
        """
        print("[yolor.detect] Running setup")
        self.lock.acquire()
        # This is probably unnecessary, but resetting flag at start of all setup methods in case it fails or something
        self._have_setup = False
        if not Path(self.output_path).exists:
            raise FileNotFoundError(self.output_path)

        source = self.source_path
        self.webcam_source = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # Load classes
        if self.class_names is None or len(self.class_names) < 1:
            with open(self.class_names_file, "r") as f:
                names = f.read().replace("\r\n","\n").split("\n")
            # Filter empties
            self.class_names = list(filter(None, names))
        self._have_setup = True
        self.lock.release()

    def display(self, text: str, ignore_verbose = False):
        """
        Display text only if in verbose mode
        """
        # @todo: attach to a logging instance or something better than this
        if not self.mode_verbose and not ignore_verbose:
            return
        print("[yolor.detect] " + text)


    def get_bounding_box_colours(self) -> list:
        """
        Return list of colours (key = class) for bounding boxes/labels
        """

        return self._class_bounding_box_colours

    def load_model(self):
        """
        Load a model (weights/config pair) into memory for use
        """
        print("[yolor.detect] Running load_model")

        self.unload_model()
        if not self._have_init_device:
            print("Init device first")
            return

        if len(self.model_weights) < 1 or not self.model_config:
            print("No weights/config specified")
            return
        # Load model
        logging.debug(f"Attempting to load model. Config: {self.model_config}, Weights: {self.model_weights}")
        model = Darknet(self.model_config, self.inference_size).cuda()
        model.load_state_dict(torch.load(self.model_weights, map_location=self._device)['model'])
        #model = attempt_load(weights, map_location=device)  # load FP32 model
        # @todo: might be worth enabling the check
        #inference_size = check_img_size(inference_size, s=model.stride.max())  # check img_size
        model.to(self._device).eval()
        if self._half_mode:
            model.half()  # to FP16
        
        self._model = model

        self._model_name_loaded = Path(self.model_weights).name.replace('.pt', '')
        if not self.model_name:
            self.model_name = self._model_name_loaded
        
        disp_string = f"Using model {self.model_name}"
        logging.debug(disp_string)
        self._have_loaded_model = True

    def unload_model(self):
        """
        Unload the model
        """
        print("[yolor.detect] Running unload_model")
        if self._help_garbage_man and self._model is not None:
            del self._model
        self._have_loaded_model, self._model, self._model_name_loaded = False, None, None

    def deinit_device(self):
        """
        De-initialise device
        """
        print("[yolor.detect] Running deinit_device")
        self.unload_model()
        self._have_init_device = False
        if self._help_garbage_man and self._device is not None:
            del self._device

    def init_device(self, no_half = False):
        """
        Initialise the device for usage
        """
        print("[yolor.detect] Running init_device")
        self.deinit_device()
        if not self._have_setup:
            print("Run setup first")
            return
        if not self.target_device:
            print("Target device not set")
            return

        self._device = select_device(self.target_device)
        self._half_mode = False

        if not no_half:
            self._half_mode = self._device.type != 'cpu'  # half precision only supported on CUDA

        self._have_init_device = True
        print(f"Device initialised: {self._device}")

    def inference(self):
        """
        Run inference according to setup. The main event.
        """
        
        print("[yolor.detect] Running inference")
        if not self._have_setup:
            print("Run setup first.")
            return

        if not self._have_init_device:
            self.init_device()

        if not self._device:
            print("Device not initialised")
            return

        if not self._have_loaded_model:
            self.load_model()

        if not self._model:
            print("Model not loaded.")
            return

        model = self._model

        source, output_path, classes_restrict = \
            self.source_path, self.output_path, self.classes_restrict

        inference_size, capture_nth_frame, names, conf_thres, iou_thres = \
            self.inference_size, self.capture_nth_frame, self.class_names, self.confidence_threshold, self.iou_threshold

        augment, agnostic_nms, display_bb, display_info, save_frames, save_txt = \
            self.mode_augment, self.nms_is_agnostic, self.display_bounding_boxes, self.display_stats, self.save_video_frames, self.save_text

        save_img, view_img = self.save_images, self.display_images

        webcam_source = self.webcam_source
        verbose = self.mode_verbose
        half = self._half_mode

        save_nth_frame = self.save_nth_frame

        if verbose:
            print("Running in verbose mode")
            
        if webcam_source:
            save_path = str(Path(output_path).joinpath('webcam_output.mp4'))
            if verbose:
                print(f"Using webcam as source, save_path: {save_path}")
        else:
            save_path = str(Path(output_path))
            if verbose:
                print(f"Saving images to path {save_path}")

        device = self._device
        if verbose:
            print(f"Using device {device}")

        # Second-stage classifier
        # @todo: look into this
        classify = False
        if classify:
            logging.debug("Using second-stage classifier")
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()

        # Set Dataloader
        # @todo: refactor
        vid_path, vid_writer = None, None
        if webcam_source:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            capture_nth_frame = capture_nth_frame if capture_nth_frame > 0 else 4
            dataset = LoadStreams(source, img_size=inference_size, nth_frame=capture_nth_frame)
        else:
            dataset = LoadImages(source, img_size=inference_size, auto_size=64, print_output=False)

        # colors = [[0, 255, 0]]
        colours = self.get_bounding_box_colours()

        stats_times = []
        stats_images = 0
        stats_detections = 0

        stats_bottom_base = [
            f"yolor-edge run: {self.run_name}",
            f"Algorithm: YOLOR; Model: '{self.model_name}'; Inf. Size {inference_size}px",
            f"Thresh: Conf {conf_thres:.3f}; IoU {iou_thres:.3f}",
            f"System: {self.system_name}; Device: {device}",
        ]

        if classes_restrict:
            restricted_classes = []
            for cls in classes_restrict:
                # @todo: not sure why "if cls in names:"" doesn't work here?
                if cls <= len(names):
                    restricted_classes.append(names[cls])
                else:
                    restricted_classes.append(str(cls))

            if len(restricted_classes):
                stats_bottom_base.append("; ".join(restricted_classes))

        stats_bottom_base = "\n".join(list(filter(None,stats_bottom_base)))

        # @todo: cleanup this chaotic mess of variables.
        frames_counted = 0
        detect_count = 0
        avg_conf = inst_fps = run_time  = 0
        video_src_width = video_src_height = 0
        running_classes = []
        running_conf = []
        running_names = []
        running_detect_count = source_frame_count = running_frame_count = 0
        prev_frame = 0
        video_resize_factor = 1
        video_resize = False
        videos_to_resize = []
        iteration_start = time.time()
        last_frame_check = None
        text_scale_factor = 0
        source_name = None
        video_mode = False

        base_text_size = None

        source_vid_writing = None
        frame_save_path = None
        source_number = 0
        last_frame_saved = None
        current_frame = 0

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, inference_size, inference_size), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        logging.debug("Beginning inference...")
        for path, img, im0s, vid_cap in dataset:

            # @todo: sus output_path what img, im0s, vid_cap contain
            # want to rescale if it's too wide
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            stats_images += 1

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=augment)[0]
            inference_time = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes_restrict, agnostic=agnostic_nms)
            t2 = time_synchronized()
            nms_time = t2 - inference_time
            inference_time = inference_time - t1

            stats_times.append(inference_time)
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            inst_detected_classes = []
            frames_counted += 1
            source_frame_count += 1
            running_frame_count += 1
            this_detect_count = 0

            if not webcam_source:
                current_frame = dataset.frame
                if current_frame == 0 or current_frame < prev_frame:
                    frames_counted = 0
                    frame_save_path = None
                    source_name = None
                    video_mode = not (dataset.mode == 'images')
                    source_number += 1
                    self.display("New source, resetting stats")
                    iteration_start = last_frame_check = time_synchronized()
                    source_frame_count = 0
                    source_frames = dataset.nframes
                    running_classes = []
                    inst_detected_classes = []
                    inst_detected_conf = []
                    running_conf = []
                    running_names = []
                    avg_conf = run_time = inst_fps = 0.
                    avg_nms = avg_inf = 0.
                    text_scale_factor = 0

            prev_frame = current_frame
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam_source:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                # s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                detections = det is not None and len(det)

                source_name = Path(p).name
                save_path = str(Path(output_path).joinpath(source_name))
                detect_count = len(det)
                inst_detected_conf = []
                inst_detected_classes = []

                if detections:
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    if verbose:
                        self.display("%d detections in %.3fs" % (detect_count, inference_time))
                    # Write results
                    for *xyxy, conf, cls in det:
                        this_detect_count += 1
                        stats_detections += 1
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            txt_path = str(Path(output_path) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
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
                                plot_one_box(xyxy, im0, label=label, color=colours[cls_int], line_thickness=2, text_color=[0,0,0], line_type=cv2.LINE_8)
                            #if video_resize:
                                    # im0 = cv2.resize(im0, (video_resize_width, video_resize_height), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
                else:
                    if verbose:
                        print(f"{current_frame} No detections")


                # @todo: refactor!
                # Print summary stats on image
                if (save_img or view_img) and display_info and source_frame_count > 1:
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
                    
                    if frames_counted >= 5:
                        frame_time = time_synchronized()
                        if last_frame_check is not None:
                            frame_time = frame_time - last_frame_check
                            if frame_time > 0:
                                last_frame_check = time_synchronized()
                                inst_fps = frames_counted / frame_time
                                frames_counted = 0

                    if text_scale_factor == 0:
                        text_scale_factor = 1
                        im_shape = im0.shape
                        print(im_shape)
                        text_scale_factor = (im_shape[0] * im_shape[1]) / (1920*1080)
                        
                    
                    plot_text_with_border(img=im0, starting_row = 1, starting_column=1, label = 'yolor-edge / E. Thompson / 2021', scale_factor = text_scale_factor, text_bold=True)

                    stats_top = ""
                    stats_top += "Inst.:\n"
                    stats_top += f" Detections: {detect_count:d}\n"
                    stats_top += f" Classes: {', '.join(inst_detected_names)}\n"
                    stats_top += f" Avg. Conf: {inst_avg_conf*100:.2f}%\n"
                    if inst_fps > 0:
                        stats_top += f" FPS: {inst_fps:.2f}\n"
                    else:
                        stats_top += f" FPS: (calc)\n"
                    stats_top += f" Inf. Time: {(1E3 * inference_time):.3f}ms\n"
                    stats_top += f" NMS Time: {(1E3 * nms_time):.3f}ms\n"
                    plot_text_with_border(img=im0, starting_row=4, starting_column=2, label = stats_top, scale_factor=text_scale_factor)
                    stats_bottom = ""
                    if video_mode:
                        source_string = f"Source: '{source_name}'"
                        if video_src_width > 0:
                            source_string += f" {video_src_width}x{video_src_height}"
                        stats_bottom += source_string + "\n"
                        stats_bottom += f"Frame: {source_frame_count}/{source_frames}\n"
                        stats_bottom += f"Runtime: {run_time:.2f}s\n"

                    stats_bottom += f"All Classes: {', '.join(running_names)}\n"
                    stats_bottom += f"Avg. Conf: {avg_conf*100:.2f}\n"
                    stats_bottom += stats_bottom_base
                    plot_text_with_border(img=im0, starting_row=2, starting_column=1, from_bottom = True, label = stats_bottom, scale_factor=text_scale_factor)

                # Display results
                if view_img:
                    cv2.imshow(self.run_name, im0)
                    # if cv2.waitKey(1) == ord('q'):  # q to quit
                        # raise StopIteration

                # @todo: save webcam frames
                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        if webcam_source:
                            # @todo: save as a video instead, obviuosly indivd images = very slow
                            pass
                            # cv2.imwrite(f"{save_path}-webcam_frame{source_frame_count}.png", im0)
                        else:
                            cv2.imwrite(save_path, im0)
                    else:
                        if source_vid_writing != source_number:  # new video
                            source_vid_writing = source_number
                            vid_path = save_path
                            self.display(f"New vid path: {vid_path}")
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            video_src_fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            video_src_width = vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                            video_src_height = vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                            video_src_width = int(video_src_width)
                            video_src_height = int(video_src_height)
                            vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*fourcc), video_src_fps, (video_src_width, video_src_height))
                        vid_writer.write(im0)

                if save_frames:
                    # Save the current frame to an image, if setup to do so
                    save_frame = False
                    if frame_save_path is None:
                        frame_save_path = vid_path + "-frames"
                        Path(frame_save_path).mkdir(parents=True, exist_ok=True)

                    if save_nth_frame == 1 or save_nth_frame < 0:
                        save_frame = True
                    else:
                        if save_nth_frame > 0 and (current_frame % save_nth_frame) == 0:
                            save_frame = True

                    if not save_frame and (last_frame_saved is None or current_frame < last_frame_saved):
                        save_frame = True

                    if save_frame and frame_save_path is not None:
                        self.display(f"Saving frame {current_frame}")
                        last_frame_saved = current_frame
                        frame_save_to = f"{frame_save_path}/frame-{current_frame:05d}.png"
                        cv2.imwrite(frame_save_to, im0)

        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()

        cv2.destroyAllWindows()
        if save_txt or save_img:
            print('Results saved to %s' % Path(output_path))

        print("Summary:\n\tImages: {images}\n\tDetections: {detections}\n\tAvg. Inf. Time: {avgtime:.3f}".format(images=stats_images, detections = stats_detections, avgtime = average(stats_times)))
        print('Done. (%.3fs)' % (time.time() - t0))