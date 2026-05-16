import re
import time
from pathlib import Path

import slicer
import vtk
import numpy as np

from .PipelineRunner import PipelineRunner
from .VMTKSegmentationLogic import VMTKSegmentationLogic
from .SkeletonizationLogic import SkeletonizationLogic
from .ClassificationLogic import ClassificationLogic
from .Signal import Signal

class Logic:
    """
    Central logic for the NeuroCTA pipeline.
    
    This class manages the vessel analysis workflow, including segmentation (nnUNet or VMTK),
    vessel classification (SAGE or GINE), and skeletonization. It coordinates sub-logic instances, tracks pipeline progress with weighted step tracking to update progress bar, and loads module resources.

    """

    def __init__(self):

        # define relatively how long each step takes
        self.STEP_WEIGHTS = {
            "segmentation": {
                "nnunet": 10,
                "vmtk":   4,
            },
            "classification": {
                "SAGE": 5,
                "GINE": 5,
            },
            "skeletonization": {
                "vmtkCenterline": 3,
                "medialAxis":     4,
            },
        }

        self.nnUNetSegLogic = None
        self.vmtkSegLogic = VMTKSegmentationLogic()
        self.classLogic = ClassificationLogic()
        self.skelLogic = SkeletonizationLogic()

        self.sampleResourcesLoaded = Signal()

        self.colorNode = None
        self._loadTopBrainColorNode()

    # ----------
    # Resources
    # ----------

    def _loadTopBrainColorNode(self):
        moduleDir = Path(__file__).parent.parent
        labelFilePath = moduleDir / "Resources" / "Colors" / "labelmap_topbrain_ct.txt"

        baseName = "labelmap_topbrain_ct"

        # Search all color table nodes
        for node in slicer.util.getNodesByClass("vtkMRMLColorTableNode"):
            if node.GetName().startswith(baseName):
                self.colorNode = node
                break

        # Load only if none found
        if self.colorNode is None:
            if labelFilePath.exists():
                self.colorNode = slicer.util.loadColorTable(str(labelFilePath))
            else:
                print(f"NeuroCTA: Couldn't find the TopBrain colour file at {labelFilePath}")

    def loadResources(self):
        moduleDir = Path(__file__).parent.parent
        sampleDataPath = moduleDir / "Resources" / "SampleData"

        # Load the cropped contrast and baseline volumes
        contrastVol = slicer.util.loadVolume(sampleDataPath / "CTABrainContrastCropped.nrrd")
        baselingVol = slicer.util.loadVolume(sampleDataPath / "CTABrainBaselineCropped.nrrd")
        print(f"NeuroCTA: Loaded contrast and baseline volumes as '{contrastVol.GetName()}' and '{baselingVol.GetName()}.")

        # Loaded TopBrain CTA volume
        fullVolumeNode = slicer.util.loadVolume(sampleDataPath / "topcow_ct_025_0000_CTA.nii.gz")
        fullVolumeNode.SetName("topcow_ct_025_0000_CTA")
        print(f"NeuroCTA: Loaded TopBrain CTA volume as '{fullVolumeNode.GetName()}'.")

        # Load GT as labelmap, apply color node, convert to segmentation
        labelmapNode = slicer.util.loadLabelVolume(sampleDataPath / "topcow_ct_025_gt.nii.gz")

        if self.colorNode is None:
            self._loadTopBrainColorNode()

        labelmapNode.GetDisplayNode().SetAndObserveColorNodeID(self.colorNode.GetID())
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.SetName("topcow_ct_025_gt")

        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapNode, segmentationNode)
        slicer.mrmlScene.RemoveNode(labelmapNode)
        print(f"NeuroCTA: Loaded TopBrain multiclass segmentation as '{segmentationNode.GetName()}'.")

        # Load binary segmentation
        binSegmentationNode = slicer.util.loadSegmentation(sampleDataPath / "topcow_ct_024_gt-binary.seg.nrrd")
        binSegmentationNode.SetName("topcow_ct_024_gt_binary")
        print(f"NeuroCTA: Loaded TopBrain binary segmentation as '{binSegmentationNode.GetName()}'.")

        # Tell Widget to refresh
        self.sampleResourcesLoaded()

    # -------------------
    # Helpers
    # -------------------

    def _reconnect(self, signal, slot):
        """Disconnect all existing slots from signal, then connect slot."""
        try:
            signal.disconnectAll()
        except Exception:
            pass
        signal.connect(slot)
 
    def _stepProgress(self, step_key, method, fraction):
        weight = self.STEP_WEIGHTS[step_key][method]
        step_start = self._runner._completedWeight / self._runner._totalWeight
        step_end = (self._runner._completedWeight + weight) / self._runner._totalWeight
        overall = step_start + fraction * (step_end - step_start)
        if self._runner._onProgress:
            self._runner._onProgress(overall)


    # -------------------
    # Pipeline
    # -------------------

    def runPipeline(self, steps, folderId, onComplete=None, onError=None, onProgress=None):
        totalWeight = sum(self.STEP_WEIGHTS.get(cfg["type"], {}).get(cfg["method"], 1) for cfg in steps)
        self._runner = PipelineRunner(
            folderId,
            onProgress=onProgress, 
            steps=steps, 
            stepWeights=[self.STEP_WEIGHTS[cfg["type"]][cfg["method"]] for cfg in steps], 
            totalWeight=totalWeight
        )
        
        stepFns = []
        for cfg in steps:
            t = cfg["type"]
            if t == "segmentation":
                stepFns.append(("Segmentation",
                                 lambda prev, c=cfg: self.stepSegmentation(prev, c)))
            elif t == "classification":
                stepFns.append(("Classification",
                                 lambda prev, c=cfg: self.stepClassification(prev, c)))
            elif t == "skeletonization":
                stepFns.append(("Skeletonization",
                                 lambda prev, c=cfg: self.stepSkeletonization(prev, c)))

        self._runner.start(stepFns, onComplete=onComplete, onError=onError)

    # -------------------
    # Segmentation
    # -------------------

    #TODO: implement a STOP button
    def stepSegmentation(self, prevNode, cfg):
        method = cfg.get("method")
        if method == "nnunet":
            self._seg_nnunet(cfg)
        elif method == "vmtk":
            self._seg_vmtk(cfg)
        else:
            self._runner.failWith(f"Segmentation: unknown method '{method}'")
            return

    def _seg_nnunet(self, cfg):
        self._installCustomNNUNetFiles()
        nn = cfg["nnunet"]
        start = time.time()

        loss_subdir = "cldice" if nn["loss"] == "cldice" else "default"
        modelFolder = Path(__file__).parent.parent / "Resources" / "Models" / "Segmentation" / loss_subdir

        if not modelFolder.exists():
            self._runner.failWith(f"Segmentation (nnUNet): Model weights not found in `{modelFolder}`")
            return
        
        try:
            from SlicerNNUNetLib import Parameter as NNUNetParameter
            from SlicerNNUNetLib import SegmentationLogic as NNUnetSegmentationLogic

            self.nnUNetSegLogic = NNUnetSegmentationLogic()

        except ImportError:
            self._runner.failWith("Segmentation (nnUNet): Could not import modules from NNUNet extension. Try reinstalling it in the Extension Manager and restarting Slicer.")
            return
        
        self.nnUNetSegLogic.setParameter(NNUNetParameter(
            modelPath=modelFolder,
            folds=nn["folds"],
            device=nn["device"],
            stepSize=nn["stepSize"],
            disableTta=nn["disableTta"],
            nProcessPreprocessing=nn["npp"],
            nProcessSegmentationExport=nn["nps"],
            checkPointName="checkpoint_best.pth",
        ))

        current_fold = [0]
        last_pct = [0]
        n_folds = [len(nn["folds"].split(","))]

        def on_progress_info(txt):
            print(txt)
            match = re.search(r'(\d+)%', txt)
            if match:
                pct = int(match.group(1))
                if pct < last_pct[0]:
                    current_fold[0] += 1
                last_pct[0] = pct

                overall_fraction = (current_fold[0] + pct / 100.0) / n_folds[0]
                self._stepProgress("segmentation", "nnunet", overall_fraction)

        def on_complete(*args):
            segNode = self.nnUNetSegLogic.loadSegmentation() 
            segNode.SetName("SEG_NNUNet")
            segNode.CreateClosedSurfaceRepresentation()
            if self.colorNode is None:
                self._loadTopBrainColorNode()

            # Build name -> (r, g, b) lookup from color table
            colorLookup = {}
            for i in range(self.colorNode.GetNumberOfColors()):
                name = self.colorNode.GetColorName(i)
                color = [0.0, 0.0, 0.0, 0.0]
                self.colorNode.GetColor(i, color)
                colorLookup[name] = (color[0], color[1], color[2])

            for i in range(segNode.GetSegmentation().GetNumberOfSegments()):
                segmentId = segNode.GetSegmentation().GetNthSegmentID(i)
                segment = segNode.GetSegmentation().GetSegment(segmentId)
                color = colorLookup.get(segment.GetName())
                if color:
                    segment.SetColor(*color)

            shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
            segNodeItemId = shNode.GetItemByDataNode(segNode)
            shNode.SetItemAttribute(segNodeItemId, "NeuroCTA.processed", "true")

            childIds = vtk.vtkIdList()
            shNode.GetItemChildren(segNodeItemId, childIds, False)
            for i in range(childIds.GetNumberOfIds()):
                shNode.SetItemAttribute(childIds.GetId(i), "NeuroCTA.processed", "true")

            print(f"Segmentation (nnunet) took {time.time() - start:.2f} seconds")
            self._runner.advanceWith(segNode)
        
        self._reconnect(self.nnUNetSegLogic.inferenceFinished, on_complete)
        self._reconnect(self.nnUNetSegLogic.progressInfo, on_progress_info)
        self.nnUNetSegLogic.startSegmentation(cfg["inputNode"])

    def _seg_vmtk(self, cfg):
        method = cfg.get("method")
        start = time.time()
        preprocessing = cfg["vmtk_preprocessing"]
        total_substeps = 4 if preprocessing == "subtraction" else 2
        completed = [0]

        def on_substep():
            completed[0] += 1
            self._stepProgress("segmentation", method, completed[0] / total_substeps)

        try:
            node = self.vmtkSegLogic.runVMTKSegmentation(
                cfg["input1"], cfg["input2"],
                self._runner._outputFolderItemId,
                preprocessing=cfg["vmtk_preprocessing"],
                filtering=cfg["vmtk_filtering"],
                onProgress=on_substep,
            )
        except RuntimeError as e:
            self._runner.failWith(f"Segmentation (VMTK): {e}")
            return

        print(f"Segmentation (vmtk) took {time.time() - start:.2f} seconds")
        self._runner.advanceWith(node)

    # -------------------
    # Classification
    # -------------------

    def stepClassification(self, prevNode, cfg):
        method = cfg.get("method")
        if method in ("SAGE", "GINE"):
            self._classificationGraph(prevNode, cfg, method)
        else:
            self._runner.failWith(f"Classification: unknown method '{method}'")
            return
    
    def _classificationGraph(self, prevNode, cfg, method):
        start = time.time()
        inputNode = prevNode or cfg.get("fallbackNode")

        if inputNode is None:
            self._runner.failWith("Classification: no input node available.")
            return

        BASE_DIR = Path(__file__).resolve().parents[1]
        modelDir = BASE_DIR / "Resources" / "Models" / "Classification" / method

        if not modelDir.exists():
            self._runner.failWith(f"Classification: Model weights not found in `{modelDir}`")
            return

        if getattr(self.classLogic, "models", None) is None or self.classLogic.loadedModelType != method:
            try:
                self.classLogic.loadModels(modelDir, method)
            except Exception as e:
                self._runner.failWith(f"Classification: {e}")
                return
            
        labelmapNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", inputNode.GetName()
        )
        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
            inputNode, labelmapNode, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY
        )
 
        seg_vol = slicer.util.arrayFromVolume(labelmapNode)
        ijkMat = vtk.vtkMatrix4x4()
        labelmapNode.GetIJKToRASMatrix(ijkMat)
        slicer.mrmlScene.RemoveNode(labelmapNode)
 
        voxelSpacing = [
            np.linalg.norm([ijkMat.GetElement(r, c) for r in range(3)])
            for c in range(3)
        ]
 
        binary_mask = (seg_vol > 0).astype(np.uint8)

        try:
            coords, preds = self.classLogic.runClassInference(binary_mask, voxelSpacing, method)
        except RuntimeError as e:
            self._runner.failWith(str(e))
            return
 
        idx_to_label = {v: k for k, v in self.classLogic.LABEL_TO_IDX.items()}

        try:
            classifiedNode = self.classLogic.buildClassifiedSegmentation(
                binary_mask, coords, preds, ijkMat, idx_to_label, self._runner._outputFolderItemId
            )
        except RuntimeError as e:
            self._runner.failWith(str(e))
            return
 
        print(f"Classification ({method}) took {time.time() - start:.2f} seconds")
        self._runner.advanceWith(classifiedNode)


    # -------------------
    # Skeletonization
    # -------------------

    def stepSkeletonization(self, prevNode, cfg):
        method = cfg.get("method")
        if method == "vmtkCenterline":
            self._skel_vmtk_centerline(prevNode, cfg)
        elif method == "medialAxis":
            self._skel_medial_axis(prevNode, cfg)
        else:
            self._runner.failWith(f"Skeletonization: unknown method '{method}'")
            return

    def _skel_vmtk_centerline(self, prevNode, cfg):
        inputNode = prevNode or cfg.get("fallbackNode")
        method = cfg.get("method")

        if inputNode is None:
            self._runner.failWith("Skeletonization: no input node.")
            return

        def on_finished(*args):
            self._runner.advanceWith(None)
        
        self._reconnect(self.skelLogic.progressUpdated,
                        lambda c, t: self._stepProgress("skeletonization", method, c / t if t > 0 else 0))
        self._reconnect(self.skelLogic.skelFinished, on_finished)
        self.skelLogic.runVMTKCenterlineExtraction(
            segmentationNode=inputNode,
            outputFolderId=self._runner._outputFolderItemId
        )
    
    def _skel_medial_axis(self, prevNode, cfg):
        inputNode = prevNode or cfg.get("fallbackNode")
        method = cfg.get("method")
        if inputNode is None:
            self._runner.failWith("Skeletonization: no input node.")
            return
 
        def on_finished(skeletons):
            try:
                self._runner.advanceWith(skeletons)
            except RuntimeError as e:
                self._runner.failWith(f"Skeletonization: {e}")
                return
 
        self._reconnect(self.skelLogic.progressUpdated,
                        lambda c, t: self._stepProgress("skeletonization", method, c / t if t > 0 else 0))
        self._reconnect(self.skelLogic.skelFinished, on_finished)

        self.skelLogic.runMedialAxisSkel(inputNode, self._runner._outputFolderItemId)

    # -------------------
    # Install helper
    # -------------------

    def _installCustomNNUNetFiles(self):
        import shutil

        try:
            __import__("nnunetv2")
        except ImportError:
            try: 
                from SlicerNNUNetLib import InstallLogic
            except:
                self._runner.failWith("Segmentation (nnUNet): Could not import InstallLogic from NNUNet extension. Try reinstalling NNUNet it in the Extension Manager and restarting Slicer.")
                return

            installLogic = InstallLogic()
            success = installLogic.setupPythonRequirements("nnunetv2")
        
            if not success:
                self._runner.failWith("Segmentation (nnUNet): nnUNetv2 installation failed.")
                return
            
            slicer.util.pip_install("numpy<2")

        import nnunetv2

        nnunetDir = Path(nnunetv2.__file__).parent
        resourcesDir = Path(__file__).parent.parent / "Resources" / "nnUNet"

        files = {
            "nnUNetTrainer_CE_DC_CLDC.py": nnunetDir / "training" / "nnUNetTrainer" / "nnUNetTrainer_CE_DC_CLDC.py",
            "compound_cldice_loss.py": nnunetDir / "training" / "loss" / "compound_cldice_loss.py",
            "cldice_loss.py": nnunetDir / "training" / "loss" / "cldice_loss.py",
            "skeletonize.py": nnunetDir / "training" / "loss" / "skeletonize.py",
            "soft_skeleton.py": nnunetDir / "training" / "loss" / "soft_skeleton.py",
        }

        for src_name, dst in files.items():
            if not dst.exists():
                shutil.copy(resourcesDir / src_name, dst)
                print(f"NeuroCTA: Installed {src_name} → {dst}")