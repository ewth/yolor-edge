import uuid
import logging
from pathlib import Path
import yolor.detect

class DetectRun:
    """
    For initialising inference runs detecting objects with YOLOR
    """
    
    # Get jiggy with the configgy
    name = "detect-test"
    version = "0.4"

    output_path_base = "/resources/inference/yolor-edge/output"
    source_path = "/resources/sources/detect-test/next-run"
    
    yolor_model = "yolor_p6"

    inference_size = 1280

    confidence_threshold = 0.4
    iou_threshold = 0.5

    logging_path = "/resources/logs/yolor-edge"
    yolor_weights = f"/resources/weights/yolor/{yolor_model}.pt"
    yolor_config = f"/yolor-edge/yolor/cfg/{yolor_model}.cfg"
    target_device = "0"
    logging_level = logging.DEBUG
    


    # Behind the scenes, no touchy
    _run_name: str
    _run_id: str
    logger: None
    _log_path: Path
    _log_file: Path


    _detect = None

    def __init__(self):
        """
        Initialise an inference run
        """
        self.logger = None

        uuid_str = str(uuid.uuid4()).split("-")
        run_id = uuid_str.pop()
        self._run_id = run_id
        run_name = f"{self.name}-{self.version}-{run_id}"
        self._run_name = run_name

        general_path_name = f"{self.name}_{self.version}"

        logger = self.get_logger()

        logger.info("Initialising YOLOR Detection...")
        logger.info(f"Setting up run {run_name}...")


        output_path = Path(self.output_path_base).joinpath(general_path_name).joinpath(run_id)
        if not output_path.exists():
            output_path.mkdir(parents=True)

        self._detect = yolor.detect.Detect(
            output_path=output_path,
            source_path=self.source_path,
            target_device=self.target_device,
            run_name = self._run_name,
            model_weights = self.yolor_weights,
            model_config = self.yolor_config,
            inference_size=self.inference_size,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold
        )
        logger.info("yolor.detect initialised")

    def get_logger(self, force_new = False) -> logging.Logger:
        """
        Return existing or instantiate new Logger
        """
        if force_new or not self.logger:
            log_path = Path(self.logging_path)
            if not log_path.exists():
                log_path.mkdir(parents=True, exist_ok=True)

            log_file = log_path.joinpath(f"run-{self._run_id}.log")
            logger = logging.getLogger(f"yolor-edge")

            # Damn the logging cookbook is cool
            # https://docs.python.org/3/howto/logging-cookbook.html#logging-cookbook
            logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(str(log_file))
            fh.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            logger.addHandler(fh)
            logger.addHandler(ch)
            logger.info(f"Logging to file {str(log_file)}")
            logger.debug(f"Logging has been setup for console and file {str(log_file)}")
            self._log_path = log_path
            self._log_file = log_file
            self.logger = logger
        return self.logger 

    def go(self):
        """
        Start the run
        """
        if self._detect is None:
            logging.critical("Detect class not loaded properly?")
            return

        logging.info(f"Starting run {self._run_name}. Get ready for launch...")

        # @todo: this may go pear-shaped with all detection's multithreading. Cross your fingers.
        result = self._detect.inference()
        print(result)


if __name__ == "__main__":
    # @todo: may need to thread lock etc here
    det = DetectRun()
    det.go()

