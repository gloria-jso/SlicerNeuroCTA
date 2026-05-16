import qt
import slicer
import ctk
import vtk
from pathlib import Path
from .Logic import Logic
import csv


class Widget(qt.QWidget):
    """
    Widget for the NeuroCTA module.
    
    Organizes the interface into three main sections:
    - Display: Subject Hierarchy tree filtered by NeuroCTA-processed cases
    - Pipeline: Configuration for segmentation, classification, and skeletonization steps
    - Export: Export coordinates and features to CSV
    
    The widget uses Qt signals and slots for responsive UI state management
    based on pipeline configuration changes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.logic = Logic()

        # For export case selector combobox
        self._shObserverTag = None 
        
        # Setup main layout
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._setupDisplayCollapsible(layout)
        self._setupPipelineCollapsible(layout)
        self._setupExportCollapsible(layout)
        
        layout.addStretch()

        # Setup observers and connections
        self._setupObservers()
        self._setupConnections()
        self._setupInitialState()

        print("\n" + "=" * 100 + "\nNeuroCTA setup complete.\nUse the console command `slicer.util.getModuleLogic('NeuroCTA').loadResources()` to load SampleData\n" + "=" * 100)

    # UI Setup Methods

    def _setupDisplayCollapsible(self, parentLayout):
        displayCollapsible = ctk.ctkCollapsibleButton()
        displayCollapsible.text = "Display"
        parentLayout.addWidget(displayCollapsible)

        displayLayout = qt.QVBoxLayout(displayCollapsible)
        displayLayout.setContentsMargins(12, 12, 12, 12)

        self.showProcessedOnlyCheckBox = qt.QCheckBox("Show processed cases only")
        self.showProcessedOnlyCheckBox.checked = True
        displayLayout.addWidget(self.showProcessedOnlyCheckBox)

        self.shTreeView = slicer.qMRMLSubjectHierarchyTreeView()
        self.shTreeView.setMRMLScene(slicer.mrmlScene)
        self.shTreeView.setColumnHidden(5, True)
        self.shTreeView.setColumnHidden(4, True)
        self.shTreeView.nodeTypes = [
            "vtkMRMLSegmentationNode",
            "vtkMRMLScalarVolumeNode",
            "vtkMRMLModelNode",
            "vtkMRMLMarkupsFiducialNode",
            "vtkMRMLFolderDisplayNode",
            "vtkMRMLTableNode",
            "vtkMRMLMarkupsCurveNode"
        ]
        self.shTreeView.header().setVisible(True)
        self.shTreeView.setSortingEnabled(True)
        self.shTreeView.sortByColumn(0, qt.Qt.AscendingOrder)

        # Make the Node table expandable
        resizableFrame = ctk.ctkExpandableWidget()
        resizableFrame.orientations = qt.Qt.Vertical
        resizableFrameLayout = qt.QVBoxLayout(resizableFrame)
        resizableFrameLayout.setContentsMargins(0, 0, 0, 3)

        splitter = qt.QSplitter(qt.Qt.Vertical)
        splitter.addWidget(self.shTreeView)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setHandleWidth(10)

        resizableFrameLayout.addWidget(splitter)
        displayLayout.addWidget(resizableFrame)

    def _setupPipelineCollapsible(self, parentLayout):
        pipelineCollapsible = ctk.ctkCollapsibleButton()
        pipelineCollapsible.text = "Pipeline"
        pipelineCollapsible.collapsed = False
        parentLayout.addWidget(pipelineCollapsible)

        pipelineLayout = qt.QVBoxLayout(pipelineCollapsible)
        pipelineLayout.setContentsMargins(12, 12, 12, 12)
        pipelineLayout.setSpacing(6)

        self._setupInputType(pipelineLayout)
        self._setupSegmentation(pipelineLayout)
        self._setupClassification(pipelineLayout)
        self._setupSkeletonization(pipelineLayout)
        self._setupOutputCase(pipelineLayout)
        self._setupApplyButton(pipelineLayout)

    def _setupInputType(self, layout):
        """Setup Input Type selector (Unsegmented Volume or Segmentation)."""
        inputsFormLayout = qt.QFormLayout()
        layout.addLayout(inputsFormLayout)

        self.unsegmentedRadio = qt.QRadioButton("Unsegmented Volume")
        self.segmentationRadio = qt.QRadioButton("Segmentation")
        self.unsegmentedRadio.setChecked(True)

        self.inputTypeGroup = qt.QButtonGroup()
        self.inputTypeGroup.setExclusive(True)
        self.inputTypeGroup.addButton(self.unsegmentedRadio)
        self.inputTypeGroup.addButton(self.segmentationRadio)

        inputTypeLayout = qt.QHBoxLayout()
        inputTypeLayout.addSpacing(12)
        inputTypeLayout.addWidget(self.unsegmentedRadio)
        inputTypeLayout.addSpacing(12)
        inputTypeLayout.addWidget(self.segmentationRadio)
        inputTypeLayout.addStretch()
        inputsFormLayout.addRow("Input Type:", inputTypeLayout)

    def _setupSegmentation(self, layout):
        """Setup Segmentation section with nnUNet and VMTK options."""
        self.segCheckBox = qt.QCheckBox("Segmentation")
        self.segCheckBox.setChecked(False)
        layout.addWidget(self.segCheckBox)

        self.segSubWidget = qt.QWidget()
        segSubLayout = qt.QVBoxLayout(self.segSubWidget)
        segSubLayout.setContentsMargins(20, 0, 0, 0)
        segSubLayout.setSpacing(4)

        self._setupVMTKSegmentation(segSubLayout)
        self._setupNNUNetSegmentation(segSubLayout)

        layout.addWidget(self.segSubWidget)

    def _setupVMTKSegmentation(self, layout):
        """Setup VMTK Segmentation options."""
        self.vmtkRadio = qt.QRadioButton("Binary VMTK Segmentation")
        self.vmtkRadio.setChecked(True)
        layout.addWidget(self.vmtkRadio)

        self.vmtkFrame = qt.QFrame()
        self.vmtkFrame.setFrameShape(qt.QFrame.StyledPanel)
        self.vmtkFrame.setFrameShadow(qt.QFrame.Raised)
        vmtkFrameLayout = qt.QFormLayout(self.vmtkFrame)
        vmtkFrameLayout.setContentsMargins(8, 8, 8, 8)
        vmtkFrameLayout.setSpacing(6)

        # Preprocessing
        self.subtractionRadio = qt.QRadioButton("Subtraction")
        self.noPreprocessingRadio = qt.QRadioButton("None")
        self.noPreprocessingRadio.setChecked(True)

        self.preprocessingGroup = qt.QButtonGroup()
        self.preprocessingGroup.addButton(self.subtractionRadio)
        self.preprocessingGroup.addButton(self.noPreprocessingRadio)

        preprocessingContainer = qt.QWidget()
        preprocessingLayout = qt.QHBoxLayout(preprocessingContainer)
        preprocessingLayout.setContentsMargins(10, 0, 0, 0)
        preprocessingLayout.addWidget(self.noPreprocessingRadio)
        preprocessingLayout.addWidget(self.subtractionRadio)
        vmtkFrameLayout.addRow("Preprocessing:", preprocessingContainer)

        # Input Volume 1
        self.inputVol1Selector = slicer.qMRMLNodeComboBox()
        self.inputVol1Selector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputVol1Selector.showChildNodeTypes = False
        self.inputVol1Selector.addEnabled = False
        self.inputVol1Selector.removeEnabled = False
        self.inputVol1Selector.setMRMLScene(slicer.mrmlScene)

        inputVol1Container = qt.QWidget()
        inputVol1Layout = qt.QHBoxLayout(inputVol1Container)
        inputVol1Layout.setContentsMargins(0, 0, 0, 0)
        self.inputVol1Label = qt.QLabel("Contrast Volume:")
        self.inputVol1Label.setFixedWidth(110)
        inputVol1Layout.addWidget(self.inputVol1Label)
        inputVol1Layout.addWidget(self.inputVol1Selector)
        vmtkFrameLayout.addRow(inputVol1Container)

        # Input Volume 2
        self.inputVol2Widget = qt.QWidget()
        inputVol2Layout = qt.QFormLayout(self.inputVol2Widget)
        inputVol2Layout.setContentsMargins(0, 0, 0, 0)
        self.inputVol2Selector = slicer.qMRMLNodeComboBox()
        self.inputVol2Selector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputVol2Selector.showChildNodeTypes = False
        self.inputVol2Selector.addEnabled = False
        self.inputVol2Selector.removeEnabled = False
        self.inputVol2Selector.setMRMLScene(slicer.mrmlScene)
        inputVol2Layout.addRow("Baseline Volume:", self.inputVol2Selector)
        vmtkFrameLayout.addRow(self.inputVol2Widget)

        # Advanced settings
        self._setupVMTKAdvanced(vmtkFrameLayout)

        layout.addWidget(self.vmtkFrame)

    def _setupVMTKAdvanced(self, parentLayout):
        """Setup VMTK advanced parameters."""
        vmtkAdvancedCollapsible = ctk.ctkCollapsibleButton()
        vmtkAdvancedCollapsible.text = "Advanced"
        vmtkAdvancedCollapsible.collapsed = True
        vmtkAdvancedCollapsible.setStyleSheet("""
            ctkCollapsibleButton {
                border: none;
                background: transparent;
                padding: 0px;
                margin: 0px;
            }
        """)
        parentLayout.addRow(vmtkAdvancedCollapsible)

        advancedOuterLayout = qt.QVBoxLayout(vmtkAdvancedCollapsible)
        advancedFrame = qt.QFrame()
        advancedFrame.setObjectName("advancedFrame")
        advancedFrame.setStyleSheet("""
            QFrame#advancedFrame {
                border: 1px solid #cfcfcf;
                border-radius: 8px;
                background: transparent;
            }
        """)
        advancedOuterLayout.addWidget(advancedFrame)
        advancedLayout = qt.QFormLayout(advancedFrame)
        advancedLayout.setContentsMargins(12, 8, 12, 8)
        advancedLayout.setSpacing(6)

        # Vesselness parameters
        advancedLayout.addRow(qt.QLabel("Vesselness Filtering"))

        self.minDiameterSpinBox = qt.QSpinBox()
        self.minDiameterSpinBox.setRange(1, 100)
        self.minDiameterSpinBox.setValue(1)
        self.minDiameterSpinBox.suffix = " voxels"
        advancedLayout.addRow("Min vessel diameter:", self.minDiameterSpinBox)

        self.maxDiameterSpinBox = qt.QSpinBox()
        self.maxDiameterSpinBox.setRange(1, 100)
        self.maxDiameterSpinBox.setValue(5)
        self.maxDiameterSpinBox.suffix = " voxels"
        advancedLayout.addRow("Max vessel diameter:", self.maxDiameterSpinBox)

        self.vesselContrastSpinBox = qt.QSpinBox()
        self.vesselContrastSpinBox.setRange(0, 500)
        self.vesselContrastSpinBox.setValue(52)
        advancedLayout.addRow("Vessel contrast:", self.vesselContrastSpinBox)

        self.suppressPlatesSlider = ctk.ctkSliderWidget()
        self.suppressPlatesSlider.minimum = 0
        self.suppressPlatesSlider.maximum = 100
        self.suppressPlatesSlider.setValue(35)
        self.suppressPlatesSlider.decimals = 0
        self.suppressPlatesSlider.suffix = " %"
        advancedLayout.addRow("Suppress plates:", self.suppressPlatesSlider)

        self.suppressBlobsSlider = ctk.ctkSliderWidget()
        self.suppressBlobsSlider.minimum = 0
        self.suppressBlobsSlider.maximum = 100
        self.suppressBlobsSlider.setValue(10)
        self.suppressBlobsSlider.decimals = 0
        self.suppressBlobsSlider.suffix = " %"
        advancedLayout.addRow("Suppress blobs:", self.suppressBlobsSlider)

        advancedLayout.addItem(qt.QSpacerItem(0, 6))

        # Segmentation threshold
        advancedLayout.addRow(qt.QLabel("Segmentation"))

        self.thresholdSlider = ctk.ctkSliderWidget()
        self.thresholdSlider.minimum = 0.0
        self.thresholdSlider.maximum = 1.0
        self.thresholdSlider.setValue(0.07)
        self.thresholdSlider.decimals = 2
        self.thresholdSlider.singleStep = 0.01
        advancedLayout.addRow("Vessel threshold:", self.thresholdSlider)

        self.islandsCheckBox = qt.QCheckBox()
        self.islandsCheckBox.setChecked(True)
        advancedLayout.addRow("Keep largest island:", self.islandsCheckBox)

        advancedLayout.addItem(qt.QSpacerItem(0, 4))
        self.resetVMTKAdvancedButton = qt.QPushButton("Reset to defaults")
        advancedLayout.addRow(self.resetVMTKAdvancedButton)
        advancedLayout.setWidget(
            advancedLayout.rowCount() - 1,
            qt.QFormLayout.SpanningRole,
            self.resetVMTKAdvancedButton
        )

    def _setupNNUNetSegmentation(self, layout):
        """Setup nnUNet Segmentation options."""
        self.nnUNetRadio = qt.QRadioButton("Multiclass nnUNet Segmentation")
        layout.addWidget(self.nnUNetRadio)

        self.segMethodGroup = qt.QButtonGroup()
        self.segMethodGroup.addButton(self.nnUNetRadio)
        self.segMethodGroup.addButton(self.vmtkRadio)

        self.nnUNetFrame = qt.QFrame()
        self.nnUNetFrame.setFrameShape(qt.QFrame.StyledPanel)
        self.nnUNetFrame.setFrameShadow(qt.QFrame.Raised)
        nnUNetFrameLayout = qt.QFormLayout(self.nnUNetFrame)
        nnUNetFrameLayout.setContentsMargins(8, 8, 8, 8)
        nnUNetFrameLayout.setSpacing(6)

        self.nnUNetInputSelector = slicer.qMRMLNodeComboBox()
        self.nnUNetInputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.nnUNetInputSelector.showChildNodeTypes = False
        self.nnUNetInputSelector.addEnabled = False
        self.nnUNetInputSelector.removeEnabled = False
        self.nnUNetInputSelector.setMRMLScene(slicer.mrmlScene)
        nnUNetFrameLayout.addRow("Input Volume:", self.nnUNetInputSelector)

        self._setupNNUNetAdvanced(nnUNetFrameLayout)

        layout.addWidget(self.nnUNetFrame)

    def _setupNNUNetAdvanced(self, parentLayout):
        """Setup nnUNet advanced parameters."""
        nnUNetAdvancedCollapsible = ctk.ctkCollapsibleButton()
        nnUNetAdvancedCollapsible.text = "Advanced"
        nnUNetAdvancedCollapsible.collapsed = True
        nnUNetAdvancedCollapsible.setStyleSheet("""
            ctkCollapsibleButton {
                border: none;
                background: transparent;
                padding: 0px;
                margin: 0px;
            }
        """)
        parentLayout.addRow(nnUNetAdvancedCollapsible)

        advancedOuterLayout = qt.QVBoxLayout(nnUNetAdvancedCollapsible)
        advancedFrame = qt.QFrame()
        advancedFrame.setObjectName("advancedFrame")
        advancedFrame.setStyleSheet("""
            QFrame#advancedFrame {
                border: 1px solid #cfcfcf;
                border-radius: 8px;
                background: transparent;
            }
        """)
        advancedOuterLayout.addWidget(advancedFrame)
        advancedLayout = qt.QFormLayout(advancedFrame)
        advancedLayout.setContentsMargins(12, 8, 12, 8)
        advancedLayout.setSpacing(6)

        # Loss function
        self.nnUNetLossGroup = qt.QButtonGroup()
        self.defaultLossRadio = qt.QRadioButton("Default")
        self.clDiceLossRadio = qt.QRadioButton("Compound clDice")
        self.nnUNetLossGroup.addButton(self.clDiceLossRadio)
        self.nnUNetLossGroup.addButton(self.defaultLossRadio)
        self.clDiceLossRadio.setChecked(True)

        lossContainer = qt.QWidget()
        lossLayout = qt.QHBoxLayout(lossContainer)
        lossLayout.setContentsMargins(0, 0, 0, 0)
        lossLayout.addWidget(self.defaultLossRadio)
        lossLayout.addWidget(self.clDiceLossRadio)
        lossLayout.addStretch()
        advancedLayout.addRow("Loss function:", lossContainer)

        # Device
        self.nnUNetDeviceCombo = qt.QComboBox()
        self.nnUNetDeviceCombo.addItem("CPU", "cpu")
        self.nnUNetDeviceCombo.addItem("CUDA", "cuda")
        self.nnUNetDeviceCombo.setCurrentIndex(0)
        advancedLayout.addRow("Device:", self.nnUNetDeviceCombo)

        # Folds
        self.nnUNetFoldsCombo = qt.QComboBox()
        self.nnUNetFoldsCombo.addItem("Fold 0 (fast)", "0")
        self.nnUNetFoldsCombo.addItem("All folds (ensemble)", "0,1,2,3,4")
        self.nnUNetFoldsCombo.setCurrentIndex(0)
        advancedLayout.addRow("Folds:", self.nnUNetFoldsCombo)

        # Step size
        self.nnUNetStepSizeSlider = ctk.ctkSliderWidget()
        self.nnUNetStepSizeSlider.minimum = 0.1
        self.nnUNetStepSizeSlider.maximum = 1.0
        self.nnUNetStepSizeSlider.setValue(0.5)
        self.nnUNetStepSizeSlider.singleStep = 0.05
        self.nnUNetStepSizeSlider.decimals = 2
        advancedLayout.addRow("Step size:", self.nnUNetStepSizeSlider)

        # Workers
        self.nnUNetPreProcSpin = qt.QSpinBox()
        self.nnUNetPreProcSpin.setRange(1, 16)
        self.nnUNetPreProcSpin.setValue(1)
        advancedLayout.addRow("Preprocessing workers:", self.nnUNetPreProcSpin)

        self.nnUNetPostProcSpin = qt.QSpinBox()
        self.nnUNetPostProcSpin.setRange(1, 16)
        self.nnUNetPostProcSpin.setValue(1)
        advancedLayout.addRow("Export workers:", self.nnUNetPostProcSpin)

        advancedLayout.addItem(qt.QSpacerItem(0, 4))
        self.resetNNUnetAdvancedButton = qt.QPushButton("Reset to defaults")
        advancedLayout.addRow(self.resetNNUnetAdvancedButton)
        advancedLayout.setWidget(
            advancedLayout.rowCount() - 1,
            qt.QFormLayout.SpanningRole,
            self.resetNNUnetAdvancedButton
        )

    def _setupClassification(self, layout):
        """Setup Vessel Classification section."""
        self.vesselClassCheckBox = qt.QCheckBox("Binary-to-Multiclass Vessel Classification")
        self.vesselClassCheckBox.setChecked(False)
        self.vesselClassCheckBox.setEnabled(False)
        layout.addWidget(self.vesselClassCheckBox)

        self.vesselClassSubWidget = qt.QWidget()
        vesselClassSubLayout = qt.QVBoxLayout(self.vesselClassSubWidget)
        vesselClassSubLayout.setContentsMargins(20, 0, 0, 0)
        vesselClassSubLayout.setSpacing(4)

        self.vesselClassInputSelector = slicer.qMRMLNodeComboBox()
        self.vesselClassInputSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.vesselClassInputSelector.showChildNodeTypes = False
        self.vesselClassInputSelector.addEnabled = False
        self.vesselClassInputSelector.removeEnabled = False
        self.vesselClassInputSelector.setMRMLScene(slicer.mrmlScene)
        self.vesselClassInputLabel = qt.QLabel("Input Segmentation:")

        inputContainer = qt.QWidget()
        inputLayout = qt.QHBoxLayout(inputContainer)
        inputLayout.setContentsMargins(0, 0, 0, 0)
        inputLayout.addWidget(self.vesselClassInputLabel)
        inputLayout.addWidget(self.vesselClassInputSelector)
        vesselClassSubLayout.addWidget(inputContainer)

        self.sageRadio = qt.QRadioButton("SAGE")
        self.gineRadio = qt.QRadioButton("GINE")
        self.sageRadio.setChecked(True)

        self.classMethodGroup = qt.QButtonGroup()
        self.classMethodGroup.addButton(self.sageRadio)
        self.classMethodGroup.addButton(self.gineRadio)

        radioContainer = qt.QWidget()
        radioLayout = qt.QHBoxLayout(radioContainer)
        radioLayout.setContentsMargins(0, 0, 0, 0)
        radioLayout.addWidget(qt.QLabel("Method:"))
        radioLayout.addSpacing(12)
        radioLayout.addWidget(self.sageRadio)
        radioLayout.addSpacing(12)
        radioLayout.addWidget(self.gineRadio)
        radioLayout.addStretch()

        vesselClassSubLayout.addWidget(radioContainer)
        layout.addWidget(self.vesselClassSubWidget)

    def _setupSkeletonization(self, layout):
        """Setup Skeletonization section."""
        self.skelCheckBox = qt.QCheckBox("Skeletonization and Feature Extraction")
        self.skelCheckBox.setChecked(False)
        self.skelCheckBox.setEnabled(False)
        layout.addWidget(self.skelCheckBox)

        self.skelSubWidget = qt.QWidget()
        skelSubLayout = qt.QVBoxLayout(self.skelSubWidget)
        skelSubLayout.setContentsMargins(20, 0, 0, 0)
        skelSubLayout.setSpacing(4)

        self.skelInputSelector = slicer.qMRMLNodeComboBox()
        self.skelInputSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.skelInputSelector.showChildNodeTypes = False
        self.skelInputSelector.addEnabled = False
        self.skelInputSelector.removeEnabled = False
        self.skelInputSelector.setMRMLScene(slicer.mrmlScene)

        self.skelInputLabel = qt.QLabel("Input Segmentation:")

        inputContainer = qt.QWidget()
        inputLayout = qt.QHBoxLayout(inputContainer)
        inputLayout.setContentsMargins(0, 0, 0, 0)
        inputLayout.addWidget(self.skelInputLabel)
        inputLayout.addWidget(self.skelInputSelector)
        skelSubLayout.addWidget(inputContainer)

        self.medialAxisRadio = qt.QRadioButton("Medial Axis Thinning")
        self.vmtkCenterlineRadio = qt.QRadioButton("VMTK Extract Centerline")
        self.medialAxisRadio.setChecked(True)

        self.skelMethodGroup = qt.QButtonGroup()
        self.skelMethodGroup.addButton(self.medialAxisRadio)
        self.skelMethodGroup.addButton(self.vmtkCenterlineRadio)

        radioContainer = qt.QWidget()
        radioLayout = qt.QHBoxLayout(radioContainer)
        radioLayout.setContentsMargins(0, 0, 0, 0)
        radioLayout.addWidget(qt.QLabel("Method:"))
        radioLayout.addSpacing(12)
        radioLayout.addWidget(self.medialAxisRadio)
        radioLayout.addSpacing(12)
        radioLayout.addWidget(self.vmtkCenterlineRadio)
        radioLayout.addStretch()

        skelSubLayout.addWidget(radioContainer)
        layout.addWidget(self.skelSubWidget)

    def _setupOutputCase(self, layout):
        """Setup Output Case name selector."""
        outputCaseLayout = qt.QFormLayout()
        outputCaseLayout.setContentsMargins(0, 6, 0, 0)
        self.outputCaseNameEdit = qt.QLineEdit()
        self.outputCaseNameEdit.setPlaceholderText("Auto-named from first input")
        outputCaseLayout.addRow("Output Folder:", self.outputCaseNameEdit)
        layout.addLayout(outputCaseLayout)

    def _setupApplyButton(self, layout):
        """Setup Run Pipeline button."""
        self.applyButton = qt.QPushButton("▶ Run Pipeline")
        self.applyButton.connect("clicked()", self._onApply)
        layout.addWidget(self.applyButton)

    def _setupExportCollapsible(self, parentLayout):
        exportCollapsible = ctk.ctkCollapsibleButton()
        exportCollapsible.text = "Export"
        exportCollapsible.collapsed = True
        parentLayout.addWidget(exportCollapsible)

        exportLayout = qt.QVBoxLayout(exportCollapsible)
        exportLayout.setContentsMargins(12, 12, 12, 12)
        exportLayout.setSpacing(6)

        exportFormLayout = qt.QFormLayout()
        defaultExportPath = slicer.app.defaultScenePath
        self.exportPathEdit = ctk.ctkPathLineEdit()
        self.exportPathEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.exportPathEdit.currentPath = defaultExportPath
        exportFormLayout.addRow("Export Directory:", self.exportPathEdit)

        self.exportCaseCombo = qt.QComboBox()
        exportFormLayout.addRow("Export Case:", self.exportCaseCombo)
        exportLayout.addLayout(exportFormLayout)

        self.exportCoordsCheckBox = qt.QCheckBox("Skeleton Coordinates (x, y, z)")
        self.exportCoordsCheckBox.setChecked(True)
        exportLayout.addWidget(self.exportCoordsCheckBox)

        self.exportFeaturesCheckBox = qt.QCheckBox("Segment Features (length, tortuosity, radius)")
        self.exportFeaturesCheckBox.setChecked(True)
        exportLayout.addWidget(self.exportFeaturesCheckBox)

        self.exportButton = qt.QPushButton("Export")
        self.exportButton.setEnabled(False)
        exportLayout.addWidget(self.exportButton)

    # Observer & Connection Setup 

    def _setupObservers(self):
        """Setup Subject Hierarchy observer for export case combo refresh."""
        self._shObserverTag = slicer.mrmlScene.GetSubjectHierarchyNode().AddObserver(
            slicer.vtkMRMLSubjectHierarchyNode.SubjectHierarchyItemRemovedEvent,
            lambda caller, event: self._refreshExportCaseCombo()
        )

    def _setupConnections(self):
        """Setup all Qt signal-slot connections."""
        # Logic signals - ensures widget and viewer refresh after sample data is loaded to scene
        self.logic.sampleResourcesLoaded.connect(self._onAnyInputSelectorChanged)

        # Display signals
        self.showProcessedOnlyCheckBox.connect("toggled(bool)", self._onShowProcessedOnly)

        # Input type signals
        self.unsegmentedRadio.connect("toggled(bool)", self._onInputTypeChanged)
        self.segmentationRadio.connect("toggled(bool)", self._onInputTypeChanged)

        # Segmentation signals
        self.segCheckBox.connect("toggled(bool)", self._onSegCheckChanged)
        self.nnUNetRadio.connect("toggled(bool)", self._onSegMethodChanged)
        self.vmtkRadio.connect("toggled(bool)", self._onSegMethodChanged)
        self.subtractionRadio.connect("toggled(bool)", self._onPreprocessingChanged)
        self.resetVMTKAdvancedButton.connect("clicked()", self._onResetVMTKAdvanced)
        self.resetNNUnetAdvancedButton.connect("clicked()", self._onResetNNUNetAdvanced)

        # Classification signals
        self.vesselClassCheckBox.connect("toggled(bool)", self._onVesselClassChanged)

        # Skeletonization signals
        self.skelCheckBox.connect("toggled(bool)", self._onSkelChanged)

        # Pipeline control signals
        self.segCheckBox.connect("toggled(bool)", self._updateApplyButton)
        self.vesselClassCheckBox.connect("toggled(bool)", self._updateApplyButton)
        self.skelCheckBox.connect("toggled(bool)", self._updateApplyButton)

        # Input selector signals - updates the viewer with the input selected
        self.nnUNetInputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._onAnyInputSelectorChanged)
        self.inputVol1Selector.connect("currentNodeChanged(vtkMRMLNode*)", self._onAnyInputSelectorChanged)
        self.vesselClassInputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._onAnyInputSelectorChanged)
        self.skelInputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._onAnyInputSelectorChanged)

        # Export signals
        self.exportButton.connect("clicked()", self._onExport)
        self.exportCaseCombo.connect("currentIndexChanged(int)", self._updateExportButton)

    def _setupInitialState(self):
        """Initialize UI state by triggering all update methods."""
        self._onInputTypeChanged()
        self._onShowProcessedOnly(True)
        self._onSegCheckChanged()
        self._onSegMethodChanged()
        self._onPreprocessingChanged()
        self._onVesselClassChanged()
        self._onSkelChanged()
        self._onAnyInputSelectorChanged()
        self._refreshExportCaseCombo()

    def __del__(self):
        """Cleanup observers on widget deletion."""
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        if shNode and self._shObserverTag:
            shNode.RemoveObserver(self._shObserverTag)

    # Slots: Display

    def _onShowProcessedOnly(self, checked):
        """Filter Subject Hierarchy tree by NeuroCTA-processed items."""
        proxyModel = self.shTreeView.sortFilterProxyModel()
        if checked:
            proxyModel.addItemAttributeFilter("NeuroCTA.processed", "true", True)
        else:
            proxyModel.removeItemAttributeFilter("NeuroCTA.processed", "true")

    # Slots: Input Type Changed

    def _onInputTypeChanged(self):
        """Update UI when input type (Volume vs Segmentation) changes."""
        isSegmentation = self.segmentationRadio.isChecked()

        self.segCheckBox.setEnabled(not isSegmentation)
        if isSegmentation:
            self.segCheckBox.setChecked(False)
        else:
            self.segCheckBox.setChecked(True)

        self._updateVesselClassEnabled()
        self._updateSkeletonizationEnabled()
        self._updateStepInputSelectors()
        self._onAnyInputSelectorChanged()

    def _updateStepInputSelectors(self):
        """Show/hide input selectors based on pipeline configuration."""
        segChecked = self.segCheckBox.isChecked()
        vesselClassChecked = self.vesselClassCheckBox.isChecked()

        # Vessel Classification: hide if segmentation step feeds it
        vesselClassHasPreviousStep = segChecked
        self.vesselClassInputSelector.setVisible(not vesselClassHasPreviousStep)
        self.vesselClassInputLabel.setVisible(not vesselClassHasPreviousStep)

        # Skeletonization: hide if any previous step feeds it
        skelHasPreviousStep = segChecked or vesselClassChecked
        self.skelInputLabel.setVisible(not skelHasPreviousStep)
        self.skelInputSelector.setVisible(not skelHasPreviousStep)

        self._onAnyInputSelectorChanged()

    # Slots: Segmentation

    def _onSegCheckChanged(self):
        """Update UI when Segmentation checkbox changes."""
        checked = self.segCheckBox.isChecked()
        self.segSubWidget.setVisible(checked)
        self._updateVesselClassEnabled()
        self._updateSkeletonizationEnabled()

    def _onSegMethodChanged(self):
        """Update UI when segmentation method (VMTK vs nnUNet) changes."""
        isVMTK = self.vmtkRadio.isChecked()
        self.vmtkFrame.setVisible(isVMTK)
        self.nnUNetFrame.setVisible(not isVMTK)

        self._updateVesselClassEnabled()
        self._onAnyInputSelectorChanged()

    def _onPreprocessingChanged(self):
        """Update UI when preprocessing method changes."""
        isSubtraction = self.subtractionRadio.isChecked()
        self.inputVol2Widget.setVisible(isSubtraction)
        self.inputVol1Label.setText("Contrast Volume:" if isSubtraction else "Input Volume:")

    def _onResetVMTKAdvanced(self):
        """Reset VMTK advanced parameters to defaults."""
        self.minDiameterSpinBox.setValue(1)
        self.maxDiameterSpinBox.setValue(5)
        self.vesselContrastSpinBox.setValue(52)
        self.suppressPlatesSlider.setValue(35)
        self.suppressBlobsSlider.setValue(10)
        self.thresholdSlider.setValue(0.07)
        self.islandsCheckBox.setChecked(True)

    def _onResetNNUNetAdvanced(self):
        """Reset nnUNet advanced parameters to defaults."""
        self.clDiceLossRadio.setChecked(True)
        self.nnUNetDeviceCombo.setCurrentIndex(0)
        self.nnUNetFoldsCombo.setCurrentIndex(0)
        self.nnUNetStepSizeSlider.setValue(0.5)
        self.nnUNetPreProcSpin.setValue(1)
        self.nnUNetPostProcSpin.setValue(1)

    def _updateVesselClassEnabled(self):
        """Enable/disable classification based on pipeline configuration."""
        isVolume = self.unsegmentedRadio.isChecked()
        isSegmentation = self.segmentationRadio.isChecked()
        isVMTK = self.vmtkRadio.isChecked()
        isSegChecked = self.segCheckBox.isChecked()

        enable = (isVolume and isSegChecked and isVMTK) or isSegmentation
        self.vesselClassCheckBox.setEnabled(enable)
        if not enable:
            self.vesselClassCheckBox.setChecked(False)

        if isSegmentation:
            self.vesselClassInputSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
            if not self.vesselClassInputSelector.currentNode():
                segNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
                if segNode:
                    self.vesselClassInputSelector.setCurrentNode(segNode)

    # Slots: Classification

    def _onVesselClassChanged(self):
        """Update UI when classification checkbox changes."""
        self.vesselClassSubWidget.setVisible(self.vesselClassCheckBox.isChecked())
        self.vesselClassInputSelector.setCurrentNode(
            slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        )
        self._updateStepInputSelectors()

    # Slots: Skeletonization

    def _updateSkeletonizationEnabled(self):
        """Enable/disable skeletonization based on pipeline configuration."""
        isVolume = self.unsegmentedRadio.isChecked()
        isSegChecked = self.segCheckBox.isChecked()

        enable = (isVolume and isSegChecked) or (not isVolume)
        self.skelCheckBox.setEnabled(enable)

        if not isVolume:
            self.skelCheckBox.setChecked(True)

        if not enable:
            self.skelCheckBox.setChecked(False)

    def _onSkelChanged(self):
        """Update UI when skeletonization checkbox changes."""
        self.skelSubWidget.setVisible(self.skelCheckBox.isChecked())

    # Slots: Pipeline Execution

    def _updateApplyButton(self):
        """Enable apply button if any pipeline step is selected."""
        anySelected = (
            self.segCheckBox.isChecked() or
            self.vesselClassCheckBox.isChecked() or
            self.skelCheckBox.isChecked()
        )
        self.applyButton.setEnabled(anySelected)

    def _onApply(self):
        """Run the pipeline with current configuration."""
        try:
            folderName = self.outputCaseNameEdit.text.strip()

            if not self._checkExistingCase(folderName):
                return

            shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
            folderId = shNode.GetItemByName(folderName)
            if not folderId:
                folderId = shNode.CreateFolderItem(shNode.GetSceneItemID(), folderName)

            shNode.SetItemAttribute(folderId, "NeuroCTA.processed", "true")
            self.outputCaseNameEdit.setText(folderName)

            steps = self._buildSteps()

            print("\n\nRunning pipeline...")
            self.logic.runPipeline(
                steps, folderId,
                onComplete=self._onPipelineComplete,
                onError=self._onPipelineError,
                onProgress=self._onPipelineProgress,
            )
        except Exception as e:
            slicer.util.errorDisplay(str(e))

    def _onPipelineProgress(self, fraction):
        """Update UI with pipeline progress."""
        pct = int(fraction * 100)
        self.applyButton.setText(f"Running... {pct}%")
        self.applyButton.setStyleSheet(f"""
            QPushButton {{
                text-align: center;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4a90d9,
                    stop:{fraction:.3f} #4a90d9,
                    stop:{min(fraction + 0.001, 1.0):.3f} #e8e8e8,
                    stop:1 #e8e8e8
                );
                border: 1px solid #aaa;
            }}
        """)
        self.applyButton.setEnabled(False)
        slicer.app.processEvents()

    def _onPipelineComplete(self, results):
        """Update UI when pipeline completes."""
        self.applyButton.setText("▶ Run Pipeline")
        self.applyButton.setStyleSheet("")
        self.applyButton.setEnabled(True)
        self._refreshExportCaseCombo()
        slicer.util.showStatusMessage("Pipeline complete.", 3000)

    def _onPipelineError(self, label):
        """Update UI when pipeline encounters error."""
        self.applyButton.setText("▶ Run Pipeline")
        self.applyButton.setStyleSheet("")
        self.applyButton.setEnabled(True)
        slicer.util.showStatusMessage("Pipeline failed.", 8000)
        slicer.util.errorDisplay(f"Pipeline failed. {label}.")

    # Slots: Input Selection & Visualization 

    def _onAnyInputSelectorChanged(self):
        """Update visualization and output folder name when input changes."""
        node = self._getFirstInputNode()

        if node:
            self.outputCaseNameEdit.setText("CASE_" + node.GetName())
            self._hideAllInputs()
            self._showSelectedNode(node)
        else:
            self.outputCaseNameEdit.clear()

    def _hideAllInputs(self):
        """Hide all volumes and segmentations in scene."""
        allVolumes = slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
        allVolumes.InitTraversal()
        v = allVolumes.GetNextItemAsObject()
        while v:
            v.SetDisplayVisibility(0)
            v = allVolumes.GetNextItemAsObject()

        allSegs = slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode")
        allSegs.InitTraversal()
        s = allSegs.GetNextItemAsObject()
        while s:
            s.SetDisplayVisibility(0)
            s = allSegs.GetNextItemAsObject()

    def _showSelectedNode(self, node):
        """Show selected node in 3D view with appropriate rendering."""
        if isinstance(node, slicer.vtkMRMLScalarVolumeNode):
            if not node.GetDisplayNode():
                node.CreateDefaultDisplayNodes()

            vrLogic = slicer.modules.volumerendering.logic()
            displayNode = vrLogic.GetFirstVolumeRenderingDisplayNode(node)
            if not displayNode:
                vrLogic.CreateDefaultVolumeRenderingNodes(node)
                displayNode = vrLogic.GetFirstVolumeRenderingDisplayNode(node)

            presetNode = vrLogic.GetPresetByName("CT-Chest")
            if presetNode and displayNode:
                displayNode.GetVolumePropertyNode().Copy(presetNode)
                displayNode.SetVisibility(True)

        if isinstance(node, slicer.vtkMRMLSegmentationNode):
            if not node.GetSegmentation().ContainsRepresentation("Closed surface"):
                node.CreateClosedSurfaceRepresentation()

        node.SetDisplayVisibility(1)

        # Fit 3D view
        bounds = [0] * 6
        node.GetRASBounds(bounds)
        threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()
        renderer = threeDView.renderWindow().GetRenderers().GetFirstRenderer()
        renderer.ResetCamera(bounds)
        threeDView.renderWindow().Render()

    def _getFirstInputNode(self):
        """Get first active input node from pipeline selectors."""
        if self.segCheckBox.isChecked():
            if self.nnUNetRadio.isChecked():
                node = self.nnUNetInputSelector.currentNode()
                if node:
                    return node
            else:
                node = self.inputVol1Selector.currentNode()
                if node:
                    return node

        if self.vesselClassCheckBox.isChecked():
            if not self.segCheckBox.isChecked():
                node = self.vesselClassInputSelector.currentNode()
                if node:
                    return node

        if self.skelCheckBox.isChecked():
            if not self.segCheckBox.isChecked() and not self.vesselClassCheckBox.isChecked():
                node = self.skelInputSelector.currentNode()
                if node:
                    return node

        return None

    # Slots: Export

    def _updateExportButton(self, index):
        """Enable/disable export button based on selection."""
        self.exportButton.setEnabled(index >= 0 and self.exportCaseCombo.currentData is not None)

    def _onExport(self):
        """Export skeleton coordinates and/or features to CSV."""
        if not self.exportCoordsCheckBox.isChecked() and not self.exportFeaturesCheckBox.isChecked():
            slicer.util.errorDisplay("Select at least one export type.")
            return

        folderId = self.exportCaseCombo.currentData
        if not folderId:
            slicer.util.errorDisplay("Select a case to export.")
            return

        dirPath = self.exportPathEdit.currentPath
        if not dirPath:
            slicer.util.errorDisplay("Select an export path.")
            return

        caseName = self.exportCaseCombo.currentText
        exportedFiles = []

        if self.exportCoordsCheckBox.isChecked():
            path = self._exportCoordinatesFromFolder(folderId, dirPath, caseName)
            if path:
                exportedFiles.append(path)

        if self.exportFeaturesCheckBox.isChecked():
            path = self._exportFeaturesFromFolder(folderId, dirPath, caseName)
            if path:
                exportedFiles.append(path)

        if exportedFiles:
            msg = "Files saved:\n" + "\n".join(exportedFiles)
            qt.QMessageBox.information(self, "Export Complete", msg)
            print(msg)

    def _refreshExportCaseCombo(self):
        """Populate export case combo with completed NeuroCTA cases."""
        self.exportCaseCombo.clear()
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        sceneItemId = shNode.GetSceneItemID()

        childIds = vtk.vtkIdList()
        shNode.GetItemChildren(sceneItemId, childIds, False)

        for i in range(childIds.GetNumberOfIds()):
            itemId = childIds.GetId(i)
            if shNode.GetItemAttribute(itemId, "NeuroCTA.processed") != "true":
                continue
            name = shNode.GetItemName(itemId)
            self.exportCaseCombo.addItem(name, itemId)

        hasItems = self.exportCaseCombo.count > 0
        self.exportCaseCombo.setEnabled(hasItems)
        self.exportButton.setEnabled(hasItems)
        self._updateExportButton(self.exportCaseCombo.currentIndex)

    def _exportCoordinatesFromFolder(self, folderId, dirPath, caseName):
        """Export skeleton coordinates to CSV."""
        filePath = f"{dirPath}/{caseName}_coordinates.csv"
        if not filePath.endswith(".csv"):
            filePath += ".csv"

        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()

        # Find Skeletons subfolder
        childIds = vtk.vtkIdList()
        shNode.GetItemChildren(folderId, childIds, False)
        skeletonsFolderId = None
        for i in range(childIds.GetNumberOfIds()):
            itemId = childIds.GetId(i)
            if shNode.GetItemAttribute(itemId, "NeuroCTA.dataType") == "Skeletons":
                skeletonsFolderId = itemId
                break

        if skeletonsFolderId is None:
            slicer.util.errorDisplay("No Skeletons folder found in selected case. Run pipeline first.")
            return

        allItems = vtk.vtkIdList()
        shNode.GetItemChildren(skeletonsFolderId, allItems, True)

        with open(filePath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["segment", "x", "y", "z"])

            for i in range(allItems.GetNumberOfIds()):
                itemId = allItems.GetId(i)
                node = shNode.GetItemDataNode(itemId)
                if not node:
                    continue

                if node.IsA("vtkMRMLMarkupsCurveNode"):
                    for k in range(node.GetNumberOfControlPoints()):
                        pos = [0, 0, 0]
                        node.GetNthControlPointPosition(k, pos)
                        writer.writerow([node.GetName(), pos[0], pos[1], pos[2]])

                elif node.IsA("vtkMRMLModelNode"):
                    polyData = node.GetPolyData()
                    if not polyData:
                        continue
                    points = polyData.GetPoints()
                    for k in range(points.GetNumberOfPoints()):
                        pos = points.GetPoint(k)
                        writer.writerow([node.GetName(), pos[0], pos[1], pos[2]])

        return filePath

    def _exportFeaturesFromFolder(self, folderId, dirPath, caseName):
        """Export vessel metrics table to CSV."""
        filePath = f"{dirPath}/{caseName}_features.csv"

        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()

        # Find VesselMetrics table
        allItems = vtk.vtkIdList()
        shNode.GetItemChildren(folderId, allItems, True)

        tableNode = None
        for i in range(allItems.GetNumberOfIds()):
            itemId = allItems.GetId(i)
            if shNode.GetItemAttribute(itemId, "NeuroCTA.dataType") == "VesselMetrics":
                tableNode = shNode.GetItemDataNode(itemId)
                break

        if tableNode is None:
            slicer.util.errorDisplay("No feature table found in selected case. Run pipeline first")
            return

        table = tableNode.GetTable()
        nCols = table.GetNumberOfColumns()
        nRows = table.GetNumberOfRows()

        with open(filePath, "w", newline="") as f:
            writer = csv.writer(f)

            headers = [table.GetColumn(c).GetName() for c in range(nCols)]
            writer.writerow(headers)

            for row in range(nRows):
                rowData = [table.GetColumn(c).GetValue(row) for c in range(nCols)]
                writer.writerow(rowData)

        return filePath

    # Helpers

    def _checkExistingCase(self, caseName):
        """Check if case exists and ask user to replace."""
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        existingItem = shNode.GetItemByName(caseName)
        if not existingItem:
            return True

        result = slicer.util.confirmYesNoDisplay(
            f"Case '{caseName}' already exists. Replace it?"
        )
        if result:
            shNode.RemoveItem(existingItem)
            return True

        return False

    def _buildSteps(self):
        """Build pipeline step configuration from UI state."""
        steps = []

        if self.segCheckBox.isChecked():
            steps.append({
                "type": "segmentation",
                "method": "nnunet" if self.nnUNetRadio.isChecked() else "vmtk",
                "inputNode": self.nnUNetInputSelector.currentNode(),
                "input1": self.inputVol1Selector.currentNode(),
                "input2": self.inputVol2Selector.currentNode() if self.subtractionRadio.isChecked() else None,
                "vmtk_preprocessing": "subtraction" if self.subtractionRadio.isChecked() else None,
                "nnunet": {
                    "loss": "cldice" if self.clDiceLossRadio.isChecked() else "default",
                    "folds": self.nnUNetFoldsCombo.currentData,
                    "device": self.nnUNetDeviceCombo.currentData,
                    "stepSize": self.nnUNetStepSizeSlider.value,
                    "npp": self.nnUNetPreProcSpin.value,
                    "nps": self.nnUNetPostProcSpin.value,
                    "disableTta": True,
                },
                "vmtk_filtering": {
                    "min_vessel_diameter_voxels": self.minDiameterSpinBox.value,
                    "max_vessel_diameter_voxels": self.maxDiameterSpinBox.value,
                    "vessel_contrast": self.vesselContrastSpinBox.value,
                    "suppress_plates_pct": self.suppressPlatesSlider.value,
                    "suppress_blobs_pct": self.suppressBlobsSlider.value,
                    "threshold_lower": self.thresholdSlider.value,
                    "apply_islands_cleanup": self.islandsCheckBox.isChecked
                }
            })

        if self.vesselClassCheckBox.isChecked():
            steps.append({
                "type": "classification",
                "method": "SAGE" if self.sageRadio.isChecked() else "GINE",
                "fallbackNode": self.vesselClassInputSelector.currentNode(),
            })

        if self.skelCheckBox.isChecked():
            steps.append({
                "type": "skeletonization",
                "method": "medialAxis" if self.medialAxisRadio.isChecked() else "vmtkCenterline",
                "fallbackNode": self.skelInputSelector.currentNode(),
            })

        return steps