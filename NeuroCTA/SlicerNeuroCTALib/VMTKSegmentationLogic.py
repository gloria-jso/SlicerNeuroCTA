import slicer
import vtk

class VMTKSegmentationLogic:
    """
    Logic for binary vessel segmentation using VMTK in 3D Slicer.
    
    This class performs vessel segmentation from CTA volumes via optional preprocessing (volume subtraction), VMTK-based vesselness filtering, and thresholding. Outputs are organized in the Subject Hierarchy and can be passed to downstream classification and skeletonization steps.
    
    Preprocessing options:
    - None: applies vesselness filtering directly to the input volume
    - Subtraction: registers and subtracts a baseline volume from contrast volume before filtering
    
    """

    def __init__(self):
        pass
        

    def runVMTKSegmentation(self, input1, input2, outputFolderId=None, preprocessing=None, filtering=None, onProgress=None):

        cfg = {**filtering}

        # Preprocessing
        if preprocessing == "subtraction":
            aligned_node = self._register_volumes(input1, input2)
            if onProgress: onProgress()

            working_node = self._subtract_volumes(aligned_node, input2)
            if onProgress: onProgress()
            
            self._addNodeToFolder(aligned_node, outputFolderId)
            self._addNodeToFolder(working_node, outputFolderId)

            # Hide original input volume
            input1.SetDisplayVisibility(False)

            # Show subtract volume
            vrLogic = slicer.modules.volumerendering.logic()

            displayNode = vrLogic.GetFirstVolumeRenderingDisplayNode(working_node)

            if not displayNode:
                vrLogic.CreateDefaultVolumeRenderingNodes(working_node)
                displayNode = vrLogic.GetFirstVolumeRenderingDisplayNode(working_node)
            
            presetNode = vrLogic.GetPresetByName("CT-Chest")
            if presetNode and displayNode:
                displayNode.GetVolumePropertyNode().Copy(presetNode)
                displayNode.SetVisibility(True)
                
        else:
            working_node = input1

        vesselness_node = self._vesselness_filtering(working_node, cfg)
        self._addNodeToFolder(vesselness_node, outputFolderId)
        if onProgress: onProgress()

        seg_node, segment_id, seg_editor_widget, _ = self._threshold_segment(
            vesselness_node, cfg
        )

        if cfg["apply_islands_cleanup"]:
            self._keep_largest_island(segment_id, seg_editor_widget)

        if onProgress: onProgress()
        seg_node.CreateClosedSurfaceRepresentation()
        return seg_node
    

    def _addNodeToFolder(self, node, folderId):
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        nodeItemId = shNode.GetItemByDataNode(node)
        shNode.SetItemParent(nodeItemId, folderId)
        shNode.SetItemAttribute(nodeItemId, "NeuroCTA.processed", "true")


    def _register_volumes(self, contrast_node, baseline_node):
        """
        Register a CTA contrast volume to a baseline volume using SlicerElastix module
        """
        
        aligned_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", "PREPROC_ContrastAligned"
        )

        try:
            elastix_logic = slicer.util.getModuleLogic("Elastix")
            elastix_logic.registerVolumes(
                baseline_node,
                contrast_node,
                outputVolumeNode=aligned_node,
            )
        except Exception as e:
            slicer.mrmlScene.RemoveNode(aligned_node)
            raise RuntimeError(f"NeuroCTA is missing Slicer Elastix extension. Install it via Extension Manager and restart Slicer")
        
        return aligned_node


    def _subtract_volumes(self, aligned_contrast_node, baseline_node):
        """
        Voxel-wise subtraction: aligned_contrast − baseline → subtracted volume using SubtractScalarVolumes CLI module.
        """
        subtracted_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "PREPROC_Subtracted")

        params = {
            "inputVolume1": aligned_contrast_node.GetID(),
            "inputVolume2": baseline_node.GetID(),
            "outputVolume": subtracted_node.GetID(),
        }

        cli_node = slicer.cli.runSync(slicer.modules.subtractscalarvolumes, None, params)

        return subtracted_node


    def _vesselness_filtering(self,subtracted_node, config: dict):
        """
        Vesselness filterning with VMTK Vesselness Filterning module
        """

        vesselness_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", "PREPROC_Vesselness"
        )
        vesselness_node.CreateDefaultDisplayNodes()

        try:
            vmtk_logic = slicer.util.getModuleLogic("VesselnessFiltering")
        except Exception as e:
            raise RuntimeError(f"NeuroCTA is missing Slicer VMTK extension. Install it via Extension Manager and restart Slicer")

        try:
            min_spacing = min(subtracted_node.GetSpacing())
            minimumDiameterMm = config["min_vessel_diameter_voxels"] * min_spacing
            maximumDiameterMm = config["max_vessel_diameter_voxels"] * min_spacing

            alpha = vmtk_logic.alphaFromSuppressPlatesPercentage(config["suppress_plates_pct"])
            beta = vmtk_logic.betaFromSuppressBlobsPercentage(config["suppress_blobs_pct"])

            vmtk_logic.computeVesselnessVolume(
                subtracted_node,
                vesselness_node,
                minimumDiameterMm=minimumDiameterMm,
                maximumDiameterMm=maximumDiameterMm,
                alpha=alpha,
                beta=beta,
                contrastMeasure=config["vessel_contrast"],
            )

        except Exception as e:
            slicer.mrmlScene.RemoveNode(vesselness_node)
            raise RuntimeError(f"Error during vesselness filtering: {e}")


        return vesselness_node


    def _threshold_segment(self, vesselness_node, config: dict):
        """
        Create a segmentation from the vesselness volume by thresholding in Segment Editor.
        """
        seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode",
        "SEG_VMTK")
        seg_node.CreateDefaultDisplayNodes()
        seg_node.SetReferenceImageGeometryParameterFromVolumeNode(vesselness_node)

        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        itemId = shNode.GetItemByDataNode(seg_node)
        shNode.SetItemAttribute(itemId, "NeuroCTA.processed", "true")

        # Add a segment
        seg = seg_node.GetSegmentation()
        segment_id = seg.AddEmptySegment("TempSegName")
        segment = seg.GetSegment(segment_id)

        # Set colour to red (arterial convention)
        segment.SetColor(0.9, 0.2, 0.2)

        # Run threshold effect via Segment Editor logic
        seg_editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        seg_editor_widget = slicer.qMRMLSegmentEditorWidget()
        seg_editor_widget.setMRMLScene(slicer.mrmlScene)
        seg_editor_widget.setMRMLSegmentEditorNode(seg_editor_node)
        seg_editor_widget.setSegmentationNode(seg_node)
        seg_editor_widget.setSourceVolumeNode(vesselness_node)
        seg_editor_widget.setCurrentSegmentID(segment_id)

        seg_editor_widget.setActiveEffectByName("Threshold")
        effect = seg_editor_widget.activeEffect()
        effect.setParameter("MinimumThreshold", str(config["threshold_lower"]))
        effect.setParameter("MaximumThreshold", "1.0")
        effect.self().onApply()

        return seg_node, segment_id, seg_editor_widget, seg_editor_node

    def _keep_largest_island(self, segment_id, seg_editor_widget):
        """Keeping the largest, using Segment Editor."""
        seg_editor_widget.setCurrentSegmentID(segment_id)
        seg_editor_widget.setActiveEffectByName("Islands")
        effect = seg_editor_widget.activeEffect()
        effect.setParameter("Operation", "KEEP_LARGEST_ISLAND")
        effect.self().onApply()
