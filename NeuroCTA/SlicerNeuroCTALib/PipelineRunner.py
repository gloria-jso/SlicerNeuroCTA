import qt
import slicer


class PipelineRunner(qt.QObject):
    """
    Performs sequential execution of pipeline steps with progress tracking.
    
    Each step is a callable that receives the output node from the previous step (or None for the first step) and is responsible for launching its computation and calling advanceWith(resultNode) when complete.
    
    Progress is tracked via weighted step completion to provide overall progress feedback.
    
    The runner manages step ordering, output node threading, and folder organization in the Subject Hierarchy.
    
    """

    def __init__(self, outputFolderItemId, onProgress=None, steps=None, stepWeights=None, totalWeight=None):
        self._steps = []
        self._currentNode = None
        self._outputFolderItemId = outputFolderItemId
        self._onProgress = onProgress
        self._stepWeights = stepWeights or {}
        self._totalWeight = totalWeight or 1
        self._completedWeight = 0
        self._stepIndex = 0
        self._onComplete = None
        self._onError = None

    def start(self, steps, onComplete=None, onError=None):
        self._steps = list(steps)
        self._onComplete = onComplete
        self._onError = onError
        self._currentNode = None
        self._stepIndex = 0
        self._completedWeight = 0
        if self._onProgress:
            self._onProgress(0.0)
        self._runNext()

    def advanceWith(self, node):
        self._currentNode = node

        if node is not None and not isinstance(node, dict):
            self._addToFolder(node)

        self._completedWeight += self._stepWeights[self._stepIndex]
        self._stepIndex += 1
        if self._onProgress:
            self._onProgress(self._completedWeight / self._totalWeight)
        self._runNext()

    def failWith(self, label):
        if self._onError:
            self._onError(label)

    def _runNext(self):
        if not self._steps:
            print("Pipeline complete!")
            slicer.util.showStatusMessage(f"Pipeline complete.")
            if self._onComplete:
                self._onComplete(self._currentNode)
            return
        label, fn = self._steps.pop(0)
        slicer.util.showStatusMessage(f"Running: {label}…")
        fn(self._currentNode)

    def _addToFolder(self, node):
        if node is None:
            return
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        itemId = shNode.GetItemByDataNode(node)
        if itemId:
            shNode.SetItemParent(itemId, self._outputFolderItemId)
            shNode.SetItemAttribute(itemId, "NeuroCTA.processed", "true")

    @property
    def outputFolderItemId(self):
        """
        Defines the ID of the folder where all the processed nodes will be saved
        """
        return self._outputFolderItemId