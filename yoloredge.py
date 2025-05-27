import uuid
import logging
from pathlib import Path
import argparse

class YolorEdge:
    """
    For initialising inference runs detecting objects with YOLOR
    """
    
    # Get jiggy with the configgy

    # Default task, unless specified otherwise
    task = "val"
    name = f"{task}-run"
    version = "0.6"

    output_path_base = f"/resources/inference/yolor-edge/{name}-output"
    output_append_run = True
    source_path = f"/resources/sources/{name}/next-run"
    # source_path = "0"
    
    yolor_model = "yolor_p6"

    inference_size = 1280

    confidence_threshold = 0.4
    iou_threshold = 0.5

    display_stats = True
    display_extra_stats = False
    display_bounding_boxes = True
    display_bounding_box_labels = True
    display_percentage_decimal = False
    mode_verbose = True
    save_video_frames = False
    save_text = False
    save_images = True
    save_nth_frame = 5
    logging_path = f"/resources/logs/yolor-edge/{name}"

    class_names_file = "/yolor-edge/data/coco-2017/coco.names"
    yolor_weights = f"/resources/weights/yolor/{yolor_model}.pt"
    yolor_config = f"/yolor-edge/yolor/cfg/{yolor_model}.cfg"
    target_device = "0"
    

    # Behind the scenes, no touchy
    _run_name: str
    _run_id: str
    _logger: None
    _log_path: Path
    _log_file: Path


    _task = None

    def __init__(self):
        """
        Initialise an inference run
        """
        uuid_str = str(uuid.uuid4()).split("-")
        run_id = uuid_str.pop()
        self._run_id = run_id
        run_name = f"{self.name}-{self.version}-{run_id}"
        self._run_name = run_name

        general_path_name = f"{self.name}_{self.version}"
        self._logger = None
        logger = self.logger()

        logger.info(f"Initialising yolor.{self.task}...")
        logger.info(f"Model: {self.yolor_model}, Weights: {self.yolor_weights}, Config: {self.yolor_config}")
        output_path = Path(self.output_path_base).joinpath(general_path_name)
        if self.output_append_run:
            output_path=output_path.joinpath(run_id)
        if not output_path.exists():
            output_path.mkdir(parents=True)
            logger.debug(f"Output path {str(output_path)} created")


        if self.task == "detect":
            self._task = self.init_detect(output_path)
        elif self.task == "val":
            self._task = self.init_validation(output_path)

        if self._task is None:
            print("No task performed.")
            logger.exception("No task performed.")
            return

        logger.debug(f"yolor.{self.task} initialised with run {self._task.run_name}")
        logger.debug("Running setup...")
        self._task.setup()
        logger.debug("Done.")

    def init_validation(self, output_path):
        import yolor.validate
        return yolor.validate.Validate(
            run_name                    = self._run_name,
            run_id = self._run_id,
            #                                                               Old Arg
            output_path                 = output_path,                      # output
            source_path                 = self.source_path,                 # source
            target_device               = self.target_device,               # device
            model_weights               = self.yolor_weights,               # weights
            model_config                = self.yolor_config,                # cfg
            inference_size              = self.inference_size,              # img-size
            confidence_threshold        = self.confidence_threshold,        # conf-thres
            iou_threshold               = self.iou_threshold,               # iou-thres
            class_names_file            = self.class_names_file,            # names
            display_bounding_boxes      = self.display_bounding_boxes,      # display-bb
            display_stats               = self.display_stats,               # display-info
            display_extra_stats         = self.display_extra_stats,
            display_bounding_box_labels = self.display_bounding_box_labels,
            display_percent_decimal     = self.display_percentage_decimal,
            save_video_frames           = self.save_video_frames,           # save-frames
            save_nth_frame              = self.save_nth_frame,              # nth-frame
            mode_verbose                = self.mode_verbose,                # verbose
            save_text                   = self.save_text,
            save_images                 = self.save_images,
            append_run_id_to_files      = False
        )

    def init_detect(self, output_path):
        import yolor.detect
        
        return yolor.detect.Detect(
            run_name                    = self._run_name,
            run_id = self._run_id,
            #                                                               Old Arg
            output_path                 = output_path,                      # output
            source_path                 = self.source_path,                 # source
            target_device               = self.target_device,               # device
            model_weights               = self.yolor_weights,               # weights
            model_config                = self.yolor_config,                # cfg
            inference_size              = self.inference_size,              # img-size
            confidence_threshold        = self.confidence_threshold,        # conf-thres
            iou_threshold               = self.iou_threshold,               # iou-thres
            class_names_file            = self.class_names_file,            # names
            display_bounding_boxes      = self.display_bounding_boxes,      # display-bb
            display_stats               = self.display_stats,               # display-info
            display_extra_stats         = self.display_extra_stats,
            display_bounding_box_labels = self.display_bounding_box_labels,
            display_percent_decimal     = self.display_percentage_decimal,
            save_video_frames           = self.save_video_frames,           # save-frames
            save_nth_frame              = self.save_nth_frame,              # nth-frame
            mode_verbose                = self.mode_verbose,                # verbose
            save_text                   = self.save_text,
            save_images                 = self.save_images,
            append_run_id_to_files      = False
        )


    def logger(self, force_new = False) -> logging.Logger:
        """
        Return existing or instantiate new Logger
        """
        if force_new or self._logger is None:
            log_path = Path(self.logging_path)
            if not log_path.exists():
                log_path.mkdir(parents=True, exist_ok=True)

            log_file = log_path.joinpath(f"run-{self._run_id}.log")
            logger = logging.getLogger(f"yolor-edge")

            # Damn the logging cookbook is cool
            # https://docs.python.org/3/howto/logging-cookbook.html#logging-cookbook
            fh = logging.FileHandler(str(log_file))
            fh.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

            logger.addHandler(fh)
            logger.debug("Logging started.")
            logger.debug(f"Logging has been setup for console and file {str(log_file)}")
            self._log_path = log_path
            self._log_file = log_file
            self._logger = logger
        return self._logger 

    def go(self):
        """
        Start the run
        """

        if self._task is None:
            self.logger().critical("Task not loaded properly?")
            return

        self.logger().info(f"Starting run {self._run_name}. Get ready for launch...")

        result = self._task.run()
        print(result)


if __name__ == "__main__":
    # @todo: may need to thread lock etc here

    parser = argparse.ArgumentParser(prog="test.py")
    parser.add_argument("--weights", nargs="+", type=str, default="yolor_p6.pt", help="model.pt path(s)")
    parser.add_argument("--data", type=str, default="/yolor-edge/data/coco-2017/coco.yaml", help="*.data path")
    parser.add_argument("--names", type=str, default="/yolor-edge/data/coco-2017/coco.names", help="*.cfg path")
    parser.add_argument("--is-coco", action="store_true", help="Indicate using COCO data")
    parser.add_argument("--batch-size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--img-size", type=int, default=768, help="inference size (pixels)")
    parser.add_argument("--log-images", type=int, default=32, help="log images")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.65, help="IOU threshold for NMS")
    parser.add_argument("--task", default="val", help=""val", "test", "study"")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a cocoapi-compatible JSON results file")
    parser.add_argument("--project", default="/resources/runs/yolor/test", help="save to project/name")
    parser.add_argument("--name", default="yolor_p6_test", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--cfg", type=str, default="/yolor-edge/yolor/cfg/yolor_p6.cfg", help="*.cfg path")
    ye = YolorEdge()
    ye.go()

