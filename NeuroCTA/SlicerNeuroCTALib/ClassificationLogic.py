# Classification Logic
import slicer
import numpy as np
import vtk

COLORS = [
    (255,  0,182), (  0,159,255), (154, 77, 66), (  0,255,190),
    (120, 63,193), ( 31,150,152), (255,172,253), (177,204,113),
    (241,  8, 92), (254,143, 66), (221,  0,255), ( 77, 62,  2),
    (255,  0,  0), (  0,255,  0), (  2,173, 36), (  0,  0,255),
    (255,255,  0), (  0,255,255), (255,  0,255), (255,239,213),
    (  0,  0,205), (205,133, 63), (210,180,140), (102,205,170),
    (  0,  0,128), (  0,139,139), ( 46,139, 87), (255,228,225),
    (106, 90,205), (221,160,221), (233,150,122), (165, 42, 42),
    (255,250,250), (147,112,219), (218,112,214), ( 75,  0,130),
    (255,182,193), ( 60,179,113), (255,235,205), (255,228,196),
]

LABEL_MAP = {
    0: "background", 1: "BA", 2: "R-P1P2", 3: "L-P1P2",
    4: "R-ICA", 5: "R-M1", 6: "L-ICA", 7: "L-M1",
    8: "R-Pcom", 9: "L-Pcom", 10: "Acom", 11: "R-A1A2",
    12: "L-A1A2", 13: "R-A3", 14: "L-A3", 15: "3rd-A2",
    16: "3rd-A3", 17: "R-M2", 18: "R-M3", 19: "L-M2",
    20: "L-M3", 21: "R-P3P4", 22: "L-P3P4", 23: "R-VA",
    24: "L-VA", 25: "R-SCA", 26: "L-SCA", 27: "R-AICA",
    28: "L-AICA", 29: "R-PICA", 30: "L-PICA", 31: "R-AChA",
    32: "L-AChA", 33: "R-OA", 34: "L-OA", 35: "VoG",
    36: "StS", 37: "ICVs", 38: "R-BVR", 39: "L-BVR", 40: "SSS"
}


class ClassificationLogic:
    """
    Logic for vessel classification using graph neural networks (GNN).
    
    This class performs multi-class classification of cerebral arteries from binary vessel segmentations.

    It uses PyTorch Geometric-based models (SAGE or GINE) to infer artery types based on skeleton topology, geometry, and connectivity features. Classifications are mapped back to the original segmentation volume via Voronoi assignment.

    """
    def __init__(self):
        self.models = None
        self.loadedModelType = None
        self.device = None

        # exclude background
        self.ARTERY_NAMES = [v for k, v in LABEL_MAP.items() if k > 0]
        self.LABEL_TO_IDX = {name: i for i, name in enumerate(self.ARTERY_NAMES)}
        self.NUM_CLASSES = len(self.ARTERY_NAMES)

    def loadModels(self, modelDir, modelType):

        try:
            import torch
            from .VesselGNN import VesselGINE, VesselSAGE
        except ImportError as e:
            raise ImportError(f"Classification: Error importing GNN models — {e}")

        self.device = torch.device("cpu")

        model_paths = sorted(modelDir.glob("*.pt"))

        if not model_paths:
            raise RuntimeError(f"No models found in {modelDir}")

        self.models = []
        for path in model_paths:
            if modelType == "SAGE":
                model = VesselSAGE(
                    in_channels=5,
                    hidden_channels=128,
                    num_classes=40,
                    n_layers=4,
                    dropout=0.3
                )
            else:
                model = VesselGINE(
                    in_channels=5,
                    edge_dim=6,
                    hidden_channels=128,
                    num_classes=40,
                    n_layers=4,
                    dropout=0.3
                )

            model.load_state_dict(torch.load(path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models.append(model)

        self.loadedModelType = modelType
        self.loadedModelDir = str(modelDir)

    def runClassInference(self, binary_mask, voxel_spacing, modelType):
        try:
            import torch
            from skimage.morphology import skeletonize
            from skan import Skeleton, summarize
            from scipy.ndimage import distance_transform_edt
        except ImportError as e:
            raise ImportError(f"Classification: missing dependency — {e}")

        spacing_zyx = [voxel_spacing[2], voxel_spacing[1], voxel_spacing[0]]

        skel = skeletonize(binary_mask)
        dist = distance_transform_edt(binary_mask, sampling=voxel_spacing)
        skel_obj = Skeleton(skel, spacing=spacing_zyx)
        stats = summarize(skel_obj, separator='-')

        coords = skel_obj.coordinates  
        degrees = skel_obj.degrees

        # Node features 
        shape_zyx = np.array(binary_mask.shape, dtype=float)
        coords_xyz = coords[:, ::-1]
        norm_coords = coords_xyz / shape_zyx[::-1]
        node_radii = dist[coords[:, 0].astype(int), coords[:, 1].astype(int), coords[:, 2].astype(int)]
        degrees_norm = degrees / (degrees.max() + 1e-6)
        node_feats = np.column_stack([norm_coords, degrees_norm, node_radii])

        # Edge features 
        edge_src, edge_dst, edge_feats = [], [], []
        for row_idx, row in stats.iterrows():
            src = int(row["node-id-src"])
            dst = int(row["node-id-dst"])
            path = skel_obj.path_coordinates(int(row_idx)).astype(int)
            radii = dist[path[:, 0], path[:, 1], path[:, 2]]
            euc = max(float(row["euclidean-distance"]), 1e-6)
            feats = [
                float(row["branch-distance"]),
                float(row["branch-distance"]) / euc,
                float(radii.mean()),
                float(radii.min()),
                float(radii.max()),
                float(radii.std()),
            ]

            edge_src.append(src)
            edge_dst.append(dst)

            edge_src.append(dst)
            edge_dst.append(src)

            edge_feats.append(feats)
            edge_feats.append(feats)

        x = torch.tensor(node_feats, dtype=torch.float)
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)


        # Inference
        with torch.no_grad():
            all_logits = []

            # loop through all model paths
            for model in self.models:
                if modelType == "SAGE":
                    out = model(x, edge_index)
                else:
                    edge_attr_np = np.array(edge_feats, dtype=np.float32)
                    mean = edge_attr_np.mean(axis=0)
                    std = edge_attr_np.std(axis=0) + 1e-6
                    edge_attr = torch.tensor((edge_attr_np - mean) / std, dtype=torch.float)
                    out = model(x, edge_index, edge_attr)

                all_logits.append(out)

            mean_logits = torch.stack(all_logits, dim=0).mean(dim=0)
            preds = mean_logits.argmax(dim=1).cpu().numpy()

        return coords, preds


    def buildClassifiedSegmentation(self, binary_mask, coords, preds, ijkToRas, idx_to_label, outputFolderId):
        """
        Voronoi-assign every foreground voxel to the nearest skeleton node,
        then create one Segment per artery class.
        """
        try:
            from scipy.spatial import cKDTree
        except ImportError as e:
            raise ImportError(f"Classification: missing dependency — {e}")

        # Voronoi assignment - nearest skeleton node per foreground voxel
        tree = cKDTree(coords)
        fg_zyx = np.argwhere(binary_mask > 0)
        _, nearest_node = tree.query(fg_zyx, workers=-1)
        fg_labels = preds[nearest_node]

        label_vol = np.full(binary_mask.shape, -1, dtype=np.int32)
        label_vol[fg_zyx[:, 0], fg_zyx[:, 1], fg_zyx[:, 2]] = fg_labels

        # Segmentation node 
        segNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", f"CLASS_{self.loadedModelType}"
        )
        segNode.CreateDefaultDisplayNodes()

        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        itemId = shNode.GetItemByDataNode(segNode)
        shNode.SetItemAttribute(itemId, "NeuroCTA.processed", "true")
        shNode.SetItemParent(itemId, outputFolderId)

        # One segment per predicted class 
        for class_idx in np.unique(fg_labels):
            if class_idx < 0:
                continue

            name = idx_to_label.get(int(class_idx), f"cls_{class_idx}")
            color = tuple(c / 255.0 for c in COLORS[int(class_idx) % len(COLORS)])

            cls_mask = (label_vol == class_idx).astype(np.uint8)
            tmpLM = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            slicer.util.updateVolumeFromArray(tmpLM, cls_mask)

            # Copy IJK→RAS geometry from the original labelmap
            ijkMat = vtk.vtkMatrix4x4()
            for r in range(4):
                for c in range(4):
                    ijkMat.SetElement(r, c, ijkToRas.GetElement(r, c))
            tmpLM.SetIJKToRASMatrix(ijkMat)

            # Import into segmentation
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                tmpLM, segNode
            )

            # Rename and recolour the segment that was just added
            seg = segNode.GetSegmentation()
            segID = seg.GetNthSegmentID(seg.GetNumberOfSegments() - 1)
            segment = seg.GetSegment(segID)
            segment.SetName(name)
            segment.SetColor(*color)

            slicer.mrmlScene.RemoveNode(tmpLM)

        segNode.CreateClosedSurfaceRepresentation()
        segNode.GetDisplayNode().SetVisibility(True)

        return segNode

