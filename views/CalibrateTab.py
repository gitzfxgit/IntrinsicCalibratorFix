from PyQt6.QtWidgets import QListView, QWidget, QGridLayout, QSizePolicy

import pyqtgraph as pg
import cv2
from pyqtgraph.parametertree import (
    Parameter,
    ParameterTree,
    RunOptions,
    InteractiveFunction,
    Interactor,
)


class CalibrateTab(QWidget):
    def __init__(self, model):
        QWidget.__init__(self)
        layout = QGridLayout()
        self.setLayout(layout)

        # create a list view from FileListModel
        self.model = model
        self.current_image = None  # Initialize current_image
        view = QListView()
        view.setModel(self.model)
        view.selectionModel().selectionChanged.connect(
            self.handle_selection_changed
        )
        layout.addWidget(view, 0, 0)

        # create image view
        self.imv = pg.ImageView()
        self.imv.show()
        layout.addWidget(self.imv, 0, 1, 0, 3)
    
        # create board view
        self.board_imv = pg.ImageView()
        self.board_imv.show()
        layout.addWidget(self.board_imv, 1, 2)

        self.parameter_detection_dict = [
            {
                'name': 'Aruco Dictionary',
                'type': 'list',
                'limits': ["4x4_1000"],
                'value': "4x4_1000",
            },
            {
                'name': 'Columns',
                'type': 'int',
                'value': "30"
            },
            {
                'name': 'Rows',
                'type': 'int',
                'value': "21"
            },
            {
                'name': 'Marker Size',
                'type': 'float',
                'value': "0.019"
            },
            {
                'name': 'Square Size',
                'type': 'float',
                'value': "0.025"
            },
            {
                'name': 'Run Detection',
                'type': 'action'
            }
        ]
        
        # Create tree of Parameter objects for Board Detection
        self.parameter_detection = Parameter.create(name='params',
                                  type='group',
                                  children=self.parameter_detection_dict)
        self.tree_detection = ParameterTree()
        self.tree_detection.setParameters(self.parameter_detection, showTop=False)
        self.parameter_detection.sigTreeStateChanged.connect(self.handle_tree_detection_changed)
        layout.addWidget(self.tree_detection, 1, 0)

        self.parameter_calibration_dict = [
            {
                'name': 'Run Calibration',
                'type': 'action'
            }
        ]
        
        # Create tree of Parameter objects for Board Detection
        self.parameter_calibration = Parameter.create(name='params',
                                  type='group',
                                  children=self.parameter_calibration_dict)
        self.tree_calibration = ParameterTree()
        self.tree_calibration.setParameters(self.parameter_calibration, showTop=False)
        self.parameter_calibration.sigTreeStateChanged.connect(self.handle_tree_calibration_changed)
        layout.addWidget(self.tree_calibration, 1, 1)

    def handle_selection_changed(self, selected, deselected):
        """
        When the selection changes, load the image into the image view
        """
        # load the image data into the ImageView
        index = selected.indexes()[0]
        image = self.model.images[index.row()]
        file_path = image.file_path
        img = cv2.imread(file_path, 0) # 0 = grayscale
        if image.board_detections:
            charuco_corners, charuco_ids, marker_corners, marker_ids = image.board_detections
            if not (marker_ids is None) and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(img, marker_corners)
            if not (charuco_ids is None) and len(charuco_ids) > 0:
                cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
        self.current_image = img
        self.imv.setImage(img.T)

    def handle_tree_detection_changed(self, tree_state, tree_change):
        """
        When the selection changes, load the image into the image view
        """
        # Only run detection when the "Run Detection" button is clicked
        if tree_change[0][0].name() != 'Run Detection':
            return
            
        print("Running detection")
        # Get values from individual parameters instead of getValues()
        aruco_dict_name = self.parameter_detection.child('Aruco Dictionary').value()
        columns = self.parameter_detection.child('Columns').value()
        rows = self.parameter_detection.child('Rows').value()
        marker_size = self.parameter_detection.child('Marker Size').value()
        square_size = self.parameter_detection.child('Square Size').value()
        
        aruco_dict = cv2.aruco.getPredefinedDictionary({
                "aruco_orig" : cv2.aruco.DICT_ARUCO_ORIGINAL,
                "4x4_250"    : cv2.aruco.DICT_4X4_250,
                "4x4_1000"    : cv2.aruco.DICT_4X4_1000,
                "5x5_250"    : cv2.aruco.DICT_5X5_250,
                "6x6_250"    : cv2.aruco.DICT_6X6_250,
                "7x7_250"    : cv2.aruco.DICT_7X7_250}[aruco_dict_name])
        print(aruco_dict_name)
        self.charuco_board = cv2.aruco.CharucoBoard((columns, rows), square_size, marker_size, aruco_dict)
        aruco_parameters = cv2.aruco.DetectorParameters()
        #aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board, detectorParams=aruco_parameters)

        # print board
        board_img = self.charuco_board.generateImage((2000, 2000))
        self.board_imv.setImage(board_img.T)

        # detect board in images
        for i, image in enumerate(self.model.images):
            img = cv2.imread(image.file_path, 0)
            image.board_detections = charuco_detector.detectBoard(img)
            charuco_corners, charuco_ids, marker_corners, marker_ids = image.board_detections
            print(f"Image {i} ({image.file_path}):")
            print(f"  Charuco corners: {charuco_corners.shape if charuco_corners is not None else None}")
            print(f"  Charuco IDs: {charuco_ids.shape if charuco_ids is not None else None}")
            print(f"  Marker corners: {len(marker_corners) if marker_corners is not None else None}")
            print(f"  Marker IDs: {marker_ids.shape if marker_ids is not None else None}")


    def handle_tree_calibration_changed(self, tree_state, tree_change):
        """
        When the selection changes, load the image into the image view
        """
        # Only run calibration when the "Run Calibration" button is clicked
        if tree_change[0][0].name() != 'Run Calibration':
            return
        
        print("Running calibration...")
        
        # Check if charuco_board exists
        if not hasattr(self, 'charuco_board'):
            print("No charuco board configured. Run detection first.")
            return
            
        # Filter out None detections and those with None corners
        detections = [(corners, ids) for image in self.model.images 
                      if image.board_detections is not None 
                      for corners, ids in [image.board_detections[:2]]
                      if corners is not None and ids is not None and len(corners) > 0]
        
        print(f"Found {len(detections)} valid detections out of {len(self.model.images)} images")
        
        if not detections:
            print("No valid board detections found. Run detection first and ensure boards are detected in images.")
            return
        
        # Get image shape from the first image
        first_image = self.model.images[0]
        img = cv2.imread(first_image.file_path, 0)
        image_shape = img.shape
            
        charuco_corners, charuco_ids = zip(*detections)
        print(f"Number of detection sets: {len(charuco_corners)}")
        for i, (corners, ids) in enumerate(zip(charuco_corners, charuco_ids)):
            print(f"  Image {i}: {len(corners)} corners, {len(ids)} ids\n")
        
        reproj_err, self.intrinsics, self.distortion, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(charuco_corners, charuco_ids, self.charuco_board, image_shape, None, None)
        print(f"intrinsics = {self.intrinsics}\n")
        print(f"distortion = {self.distortion}\n")
        print(f"rvecs = {rvecs}\n")
        print(f"tvecs= {tvecs}\n")
        print("Reprojection error: {}".format(reproj_err))