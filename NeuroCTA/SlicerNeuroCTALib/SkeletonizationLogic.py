# SkeletonizationLogic.py
import sys
import json
import numpy as np
from pathlib import Path

import time

import qt
import slicer
import vtk

from .Signal import Signal
from .Process import Process

class SkeletonizationLogic:
    """
    Logic for skeleton extraction from vessel segmentations.
    
    This class extracts 1D skeletal representations (VMTK centerlines/medial axes) from 3D segmentations and produces skeleton models, branch points, and endpoint fiducials. Two extraction methods are supported:
    - Medial Axis Thinning: extracts medial axis via external worker process
    - VMTK Extract Centerline: extracts centerlines using VMTK and Slicer model generation
    
    Signals (Qt signals for asynchronous callbacks):
        skelFinished: emitted with skeleton output dictionary when extraction completes
        errorOccurred: emitted with error message on failure
        progressInfo: emitted with progress text updates
        progressUpdated: emitted with (current, total) progress counts
    
    """

    def __init__(self):
        self.skelFinished = Signal("object")
        self.errorOccurred = Signal("str")
        self.progressInfo = Signal("str")
        self.progressUpdated = Signal("int", "int")

        self.medialAxisSkelProcess = Process(qt.QProcess.MergedChannels)
        self.medialAxisSkelProcess.finished.connect(self._onMedialAxisSkelFinished)
        self.medialAxisSkelProcess.errorOccurred.connect(self.errorOccurred)
        self.medialAxisSkelProcess.readInfo.connect(self._onMedialAxisProgressInfo)

        self._tmpDir = qt.QTemporaryDir()
        self._ijkToRas = None
        self.parentFolderId = None

        self.method = None

    @property
    def ijkToRas(self):
        return self._ijkToRas

    def __del__(self):
        self.stop()

    def stop(self):
        self.medialAxisSkelProcess.stop()

    def waitForFinished(self):
        self.medialAxisSkelProcess.waitForFinished()

    
    # -------------
    # Public
    # -------------

    def runMedialAxisSkel(self, inputNode, parentFolderId):
        self.parentFolderId = parentFolderId
        self.method = "MedialAxis"

        name = f"{inputNode.GetName()}"
        labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", name)
        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
            inputNode, labelmapNode, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY
        )

        volumeArray = slicer.util.arrayFromVolume(labelmapNode)
        self._ijkToRas = vtk.vtkMatrix4x4()
        labelmapNode.GetIJKToRASMatrix(self._ijkToRas)

        segmentation = inputNode.GetSegmentation()
        self._voxelSpacing = [
            np.linalg.norm([self._ijkToRas.GetElement(r, 0) for r in range(3)]),
            np.linalg.norm([self._ijkToRas.GetElement(r, 1) for r in range(3)]),
            np.linalg.norm([self._ijkToRas.GetElement(r, 2) for r in range(3)]),
        ]

        if not self._prepareWorkDir(volumeArray, segmentation):
            self.errorOccurred("Failed to write inputs to temp directory.")
            return

        slicer.mrmlScene.RemoveNode(labelmapNode)

        self._skelStartTime = time.time()
        self._startSkelProcess()

    
    def runVMTKCenterlineExtraction(self, segmentationNode, outputFolderId=None):
        self.method = "VMTK"
        self._segmentationNode = segmentationNode
        self._skelStartTime = time.time()

        try:
            skeletons = self._extractAllCenterlinesVMTK(outputFolderId)
            elapsed = time.time() - self._skelStartTime
            print(f"Skeletonization (vmtk) took {elapsed:.2f} seconds")
            self.skelFinished(skeletons)
        except Exception as e:
            self.errorOccurred(str(e))

    # ----------------------
    # Private - medial axis
    # ----------------------

    def _onMedialAxisSkelFinished(self, *args):
        print(f"Skeletonization (medial axis) took {time.time() - self._skelStartTime:.2f} seconds")

        try:
            skeletons = self._loadMedialAxisResults()
            self.skelFinished(skeletons)
        except RuntimeError as e:
            self.errorOccurred(f"Skeletonization: {e}")

    def _onMedialAxisProgressInfo(self, msg: str):
        for line in msg.strip().splitlines():
            if line.startswith("PROGRESS:"):
                parts = line.split("\t")
                progress, readable = parts[0], parts[1] if len(parts) > 1 else ""
                tokens = progress.split(":")
                if len(tokens) == 3:
                    _, current, total = tokens
                    self.progressUpdated(int(current), int(total))
                if readable:
                    self.progressInfo(readable)  # logs cleanly
            else:
                self.progressInfo(line)

    def _loadMedialAxisResults(self):
        output_path = Path(self._tmpDir.path()) / "results.json"
        if not output_path.exists():
            raise RuntimeError("Worker output not found.")
        
        with open(output_path) as f:
            results = json.load(f)

        m = np.array([[self._ijkToRas.GetElement(r, c) for c in range(4)] for r in range(4)])

        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        skelFolder = shNode.CreateFolderItem(self.parentFolderId, "SKEL_MedialAxis")
        bpFolder = shNode.CreateFolderItem(self.parentFolderId, "BP_MedialAxis")
        epFolder = shNode.CreateFolderItem(self.parentFolderId, "EP_MedialAxis")


        shNode.SetItemAttribute(skelFolder, "NeuroCTA.dataType", "Skeletons")
        shNode.SetItemAttribute(skelFolder, "NeuroCTA.processed", "true")
        shNode.SetItemAttribute(bpFolder, "NeuroCTA.processed", "true")
        shNode.SetItemAttribute(epFolder, "NeuroCTA.processed", "true")

        skeletonsBySegment = {}
        for segmentName, data in results.items():
            coords = np.array(data["coords"])
            r, g, b = data["color"]

            if len(coords) == 0:
                continue

            ijk_h = np.column_stack([coords[:, 2], coords[:, 1], coords[:, 0], np.ones(len(coords))])
            ras_pts = (m @ ijk_h.T).T[:, :3]

            # Skeleton model
            points = vtk.vtkPoints()
            points.SetData(vtk.util.numpy_support.numpy_to_vtk(ras_pts, deep=True))
            nPts = points.GetNumberOfPoints()
            cells = np.hstack([
                np.ones((nPts, 1), dtype=np.int64),
                np.arange(nPts, dtype=np.int64).reshape(-1, 1)
            ])
            verts = vtk.vtkCellArray()
            verts.SetCells(nPts, vtk.util.numpy_support.numpy_to_vtkIdTypeArray(cells.ravel()))

            polyData = vtk.vtkPolyData()
            polyData.SetPoints(points)
            polyData.SetVerts(verts)

            modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"Skeleton_{segmentName}")
            modelNode.SetAndObserveMesh(polyData)
            modelNode.CreateDefaultDisplayNodes()

            modelItemId = shNode.GetItemByDataNode(modelNode)
            shNode.SetItemParent(modelItemId, skelFolder)
            shNode.SetItemAttribute(modelItemId, "NeuroCTA.processed", "true")

            displayNode = modelNode.GetDisplayNode()
            displayNode.SetColor(r, g, b)
            displayNode.SetPointSize(3)
            displayNode.SetLineWidth(3)
            displayNode.SetVisibility(True)

            skeletonsBySegment[segmentName] = {
                "modelNode": modelNode,
                "features":  data.get("features", {}),
                "ep_node_id": data.get("ep_node_id")
            }

            # Branch points
            bp_coords = data.get("features", {}).get("branch_point_coords", [])
            if bp_coords:
                bp_coords = np.array(bp_coords)
                ijk_h = np.column_stack([bp_coords[:, 2], bp_coords[:, 1], bp_coords[:, 0], np.ones(len(bp_coords))])
                ras_bp = (m @ ijk_h.T).T[:, :3]
                self._addFiducialNode(f"BP_{segmentName}", ras_bp, r, g, b, bpFolder, shNode, glyphScale=2.0)

            # End points
            ep_coords = data.get("features", {}).get("end_point_coords", [])
            if ep_coords:
                ep_coords = np.array(ep_coords)
                ijk_h = np.column_stack([ep_coords[:, 2], ep_coords[:, 1], ep_coords[:, 0], np.ones(len(ep_coords))])
                ras_ep = (m @ ijk_h.T).T[:, :3]
                self._addFiducialNode(f"EP_{segmentName}", ras_ep, r, g, b, epFolder, shNode, glyphScale=2.0)

        self._populateSkeletonTable(skeletonsBySegment, self.parentFolderId)
        return skeletonsBySegment
    
     # Medial Axis Helpers
    def _prepareWorkDir(self, volumeArray, segmentation) -> bool:
        """Serialize volume array + label metadata to the temp directory."""
        tmp = Path(self._tmpDir.path())
        np.save(tmp / "volume.npy", volumeArray)

        uniqueLabels = np.unique(volumeArray)
        uniqueLabels = uniqueLabels[uniqueLabels > 0]

        labels_info = []
        for label in uniqueLabels:
            segmentId = segmentation.GetNthSegmentID(int(label) - 1)
            segment = segmentation.GetSegment(segmentId)
            r, g, b = segment.GetColor()
            labels_info.append({
                "label": int(label),
                "name":  segment.GetName(),
                "color": [r, g, b]
            })

        with open(tmp / "labels.json", "w") as f:
            json.dump(labels_info, f)

        return (tmp / "volume.npy").exists()

    def _startSkelProcess(self):
        tmp = Path(self._tmpDir.path())
        worker_path = Path(__file__).parent / "ma_skeletonize_worker.py"

        if not worker_path.exists():
            self.errorOccurred(f"Worker script not found at {worker_path}")
            return

        args_dict = {
            "volume_path": str(tmp / "volume.npy"),
            "labels_path": str(tmp / "labels.json"),
            "output_path": str(tmp / "results.json"),
            "voxel_spacing": self._voxelSpacing,
        }

        self.medialAxisSkelProcess.start(
            sys.executable,
            [str(worker_path), json.dumps(args_dict)],
            qt.QProcess.Unbuffered | qt.QProcess.ReadOnly
        )
    
    # ----------------------
    # Private - VMTK
    # ----------------------

    def _extractAllCenterlinesVMTK(self, outputFolderId):
        try:
            from ExtractCenterline import ExtractCenterlineLogic
            logic = ExtractCenterlineLogic()
        except Exception as e:
            raise RuntimeError("ExtractCenterline module not found. Install the Slicer VMTK extension via the Extensions Manager and restart Slicer.")
        
        segmentation = self._segmentationNode.GetSegmentation()
        nSegments = segmentation.GetNumberOfSegments()
        results = {}

        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        vmtkFolderId = shNode.CreateFolderItem(outputFolderId,"SKEL_VMTK")
        epFolderId = shNode.CreateFolderItem(outputFolderId, "EP_VMTK")

        shNode.SetItemAttribute(vmtkFolderId, "NeuroCTA.dataType", "Skeletons")
        shNode.SetItemAttribute(epFolderId, "NeuroCTA.processed", "true")

        for i in range(nSegments):
            segmentId = segmentation.GetNthSegmentID(i)
            segment = segmentation.GetSegment(segmentId)
            segmentName = segment.GetName()
            r, g, b = segment.GetColor()

            self.progressUpdated(i, nSegments)
            slicer.app.processEvents()

            try:
                polyData = logic.polyDataFromNode(self._segmentationNode, segmentId)
                if not polyData or polyData.GetNumberOfPoints() == 0:
                    self.progressInfo(f"Skipping {segmentName} — empty surface")
                    continue

                preprocessed = logic.preprocess(
                    polyData,
                    targetNumberOfPoints=5000,
                    decimationAggressiveness=4.0,
                    subdivide=False
                )

                # Extract network to get endpoints automatically
                networkPolyData = logic.extractNetwork(
                    preprocessed,
                    endPointsMarkupsNode=None,
                    computeGeometry=True
                )
                endpointPositions = logic.getEndPoints(networkPolyData, startPointPosition=None)

                # Create endpoint markups node
                epNode = self._addFiducialNode(
                    f"EP_{segmentName}",
                    np.array(list(endpointPositions)),
                    r, g, b, epFolderId, shNode, glyphScale=2.0
                )

                # Extract centerline
                curveSamplingDistance = 1.0
                centerlinePolyData, voronoiPolyData = logic.extractCenterline(
                    preprocessed, epNode, curveSamplingDistance
                )

                centerlinePoints = centerlinePolyData.GetPoints()
                radiusArray = centerlinePolyData.GetPointData().GetArray("Radius")

                coords_xyzr = []
                for ptIdx in range(centerlinePoints.GetNumberOfPoints()):
                    pos = centerlinePoints.GetPoint(ptIdx)
                    radius = float(radiusArray.GetValue(ptIdx)) if radiusArray else 0.0
                    coords_xyzr.append([pos[0], pos[1], pos[2], radius])

                # Network curve node
                networkCurveNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLMarkupsCurveNode", f"NetworkCurve_{segmentName}"
                )
                curveItemId = shNode.GetItemByDataNode(networkCurveNode)
                shNode.SetItemParent(curveItemId, vmtkFolderId)
                shNode.SetItemAttribute(curveItemId, "NeuroCTA.processed", "true")

                logic.addNetworkCurves(networkPolyData, networkCurveNode)
                
                # Style child curve nodes
                childIDs = vtk.vtkIdList()
                shNode.GetItemChildren(shNode.GetItemByDataNode(networkCurveNode), childIDs, True)

                for j in range(childIDs.GetNumberOfIds()):
                    childItemID = childIDs.GetId(j)
                    childNode = shNode.GetItemDataNode(childItemID)

                    shNode.SetItemAttribute(childItemID, "NeuroCTA.processed", "true")

                    if childNode and childNode.IsA("vtkMRMLMarkupsCurveNode"):
                        displayNode = childNode.GetDisplayNode()
                        if displayNode:
                            displayNode.SetColor(r, g, b)
                            displayNode.SetSelectedColor(r, g, b)

                networkCurveNode.CreateDefaultDisplayNodes()
                displayNode = networkCurveNode.GetDisplayNode()
                displayNode.SetColor(r, g, b)
                displayNode.SetSelectedColor(r, g, b)
                            
                # Metrics
                tableNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLTableNode", f"tmp_table_{segmentName}"
                )
                logic.addNetworkProperties(networkPolyData, tableNode)
                table = tableNode.GetTable()

                lengths, tortuosities, radii_vals = [], [], []

                lengthCol = table.GetColumnByName("Length")
                tortuosityCol = table.GetColumnByName("Tortuosity")
                radiusCol = table.GetColumnByName("Radius")

                # Iterate through all curves in the network
                for row in range(table.GetNumberOfRows()):

                    lengthVal = (
                        float(table.GetColumnByName("Length").GetValue(row))
                        if lengthCol else 0.0
                    )

                    tortVal = (
                        float(tortuosityCol.GetValue(row)) + 1
                        if tortuosityCol else 0.0
                    )

                    radiusVal = (
                        float(radiusCol.GetValue(row))
                        if radiusCol else 0.0
                    )

                    lengths.append(lengthVal)
                    tortuosities.append(tortVal)
                    radii_vals.append(radiusVal)
                
                slicer.mrmlScene.RemoveNode(tableNode)
                totalLength = float(sum(lengths))

                # Weight radius by branch length
                if totalLength > 0:
                    weightedMeanRadius = float(
                        sum(rad * l for rad, l in zip(radii_vals, lengths)) / totalLength
                    )
                else:
                    weightedMeanRadius = 0.0

                # Detect branching; don't show tortuosity if segment is branching
                tortuosityMetric = float(tortuosities[0]) if table.GetNumberOfRows() == 1 and tortuosities else ""

                # Collect centerline coords
                centerlinePoints = centerlinePolyData.GetPoints()

                coords_ras = [
                    list(centerlinePoints.GetPoint(ptIdx))
                    for ptIdx in range(centerlinePoints.GetNumberOfPoints())
                ]

                results[segmentName] = {
                    "coords": coords_ras,
                    "coords_xyzr": coords_xyzr,
                    "color": [r, g, b],
                    "ep_node_id": epNode.GetID(),
                    "network_curve_node_id": networkCurveNode.GetID(),
                    "features": {
                        "length": totalLength,
                        "tortuosity_dm": tortuosityMetric,
                        "mean_radius": weightedMeanRadius,
                        "min_radius": (
                            float(np.min(radii_vals))
                            if radii_vals else 0.0
                        ),
                        "max_radius": (
                            float(np.max(radii_vals))
                            if radii_vals else 0.0
                        ),
                        "branch_point_coords": [],
                        "end_point_coords": [
                            list(pos) for pos in endpointPositions
                        ],
                    }
                }

                self.progressInfo(f"[{i+1}/{nSegments}]")

            except Exception as e:
                self.progressInfo(f"Failed {segmentName}: {e}")

        self.progressUpdated(nSegments, nSegments)
        self._populateSkeletonTable(results, outputFolderId)
        return results
    
    # --------------------------
    # Private - General helpers
    # --------------------------
    
    def _addFiducialNode(self, name, ras_pts, r, g, b, folderId, shNode, glyphScale=None):
        existing = slicer.mrmlScene.GetFirstNodeByName(name)
        if existing:
            slicer.mrmlScene.RemoveNode(existing)
 
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
        node.CreateDefaultDisplayNodes()
        displayNode = node.GetDisplayNode()
        displayNode.SetSelectedColor(r, g, b)
        displayNode.SetColor(r, g, b)
        displayNode.SetTextScale(0)
        if glyphScale is not None:
            displayNode.SetGlyphScale(glyphScale)
 
        for pt in ras_pts:
            node.AddControlPoint(vtk.vtkVector3d(pt[0], pt[1], pt[2]))
 
        itemId = shNode.GetItemByDataNode(node)
        shNode.SetItemParent(itemId, folderId)
        shNode.SetItemAttribute(itemId, "NeuroCTA.processed", "true")
        return node
    
    def _populateSkeletonTable(self, skeletonsBySegment, outputFolderId):
        # Remove existing table if present
        existing = slicer.mrmlScene.GetFirstNodeByName("VesselMetrics")
        if existing:
            slicer.mrmlScene.RemoveNode(existing)

        tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", f"FEAT_{self.method}")
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        
        # Define columns
        columns = {
            "Vessel": vtk.vtkStringArray(),
            "Length (mm)": vtk.vtkStringArray(),
            "Tortuosity (DM)": vtk.vtkStringArray(),
            "Tortuosity (SOAM)": vtk.vtkStringArray(),
            "Mean Radius (mm)": vtk.vtkStringArray(),
            "Min Radius (mm)": vtk.vtkStringArray(),
            "Max Radius (mm)": vtk.vtkStringArray(),
        }
        for name, arr in columns.items():
            arr.SetName(name)
            tableNode.AddColumn(arr)

        table = tableNode.GetTable()

        def fmt(v):
            return f"{v:.3f}" if isinstance(v, float) else str(v)

        for segmentName, data in skeletonsBySegment.items():
            features = data.get("features", {})
            table.InsertNextBlankRow()
            row = table.GetNumberOfRows() - 1
            table.GetColumnByName("Vessel").SetValue(row, segmentName)
            table.GetColumnByName("Length (mm)").SetValue(row, fmt(features.get("length", 0)))
            table.GetColumnByName("Tortuosity (DM)").SetValue(row, fmt(features.get("tortuosity_dm", 0)))
            table.GetColumnByName("Tortuosity (SOAM)").SetValue(row, fmt(features.get("tortuosity_soam", 0)))
            table.GetColumnByName("Mean Radius (mm)").SetValue(row, fmt(features.get("mean_radius", 0)))
            table.GetColumnByName("Min Radius (mm)").SetValue(row, fmt(features.get("min_radius", 0)))
            table.GetColumnByName("Max Radius (mm)").SetValue(row, fmt(features.get("max_radius", 0)))

        table.Modified()

        # Add to output folder
        tableItemId = shNode.GetItemByDataNode(tableNode)
        shNode.SetItemAttribute(tableItemId, "NeuroCTA.processed", "true")
        shNode.SetItemAttribute(tableItemId, "NeuroCTA.dataType", "VesselMetrics")
        shNode.SetItemParent(tableItemId, outputFolderId)