import logging
import os
from SlicerNeuroCTALib import Widget
import qt
import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *

#
# NeuroCTA
#

class NeuroCTA(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("NeuroCTA")
        self.parent.categories = ["Segmentation", "Vascular Imaging"]
        self.parent.dependencies = [
            "Segmentations",
            "SubtractScalarVolumes",
            "VolumeRendering",
        ]
        self.parent.contributors = ["Gloria So (University of Waterloo)"]
        self.parent.helpText = _("""
NeuroCTA provides CTA vessel analysis inside 3D Slicer, including vessel segmentation, graph-based artery classification, and skeleton extraction.
Use this module to run nnUNet or VMTK segmentation, optionally classify vessels with SAGE/GINE models, and export skeleton coordinates and vessel metrics.
""")
        self.parent.acknowledgementText = _(
            ""
        )

#
# NeuroCTAWidget
#


class NeuroCTAWidget(ScriptedLoadableModuleWidget):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """
    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None
        self.widget = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        #TODO: write an InstallLogic class to do this more thoroughly like the NNUNet module.
        print("NeuroCTA: Checking dependencies...")
        slicer.app.processEvents()

        self._checkDependencies()
        self._installDependencies()

        slicer.util.showStatusMessage("")

        ScriptedLoadableModuleWidget.setup(self)
        self.widget = Widget()
        self.logic = self.widget.logic
        self.layout.addWidget(self.widget)
    
    def enter(self):
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalView)
        self.widget._onAnyInputSelectorChanged()

    def onReload(self):
        import importlib
        import importlib.util
        import sys

        packageName = "SlicerNeuroCTALib"
        submoduleNames = [
            "Signal", "VesselGNN", "Process", "VMTKSegmentationLogic",
            "PipelineRunner", "SkeletonizationLogic", "ClassificationLogic",
            "Logic", "Widget"
        ]

        # Reload package first
        if packageName in sys.modules:
            importlib.reload(sys.modules[packageName])

        # Reload each submodule
        for submoduleName in submoduleNames:
            fullName = f"{packageName}.{submoduleName}"
            print(f"Reloading {fullName}")
            if fullName in sys.modules:
                importlib.reload(sys.modules[fullName])

        ScriptedLoadableModuleWidget.onReload(self)

    def _checkDependencies(self):
        required = [
            ("elastix", "SlicerElastix"),
            ("slicernnunet", "NNUNet"),
            ("vesselnessfiltering", "SlicerVMTK"),
            ("extractcenterline",   "SlicerVMTK"),
        ]

        missing_extensions = set()
        for moduleKey, extensionName in required:
            if not hasattr(slicer.modules, moduleKey):
                missing_extensions.add(extensionName)

        if missing_extensions:
            message = (
                "NeuroCTA is missing required extension(s):\n\n- "
                + "\n- ".join(sorted(missing_extensions))
                + "\n\nInstall them via Extension Manager and restart Slicer."
            )
            slicer.util.errorDisplay(message)
        else:
            print("NeuroCTA: All required extensions were found!")

    def _installDependencies(self):        
        packages = [
            ("imageio","imageio==2.37.2"),
            ("matplotlib", "matplotlib==3.9.4"),
            ("openpyxl", "openpyxl==3.1.5"),
            ("skimage", "scikit-image==0.24.0"),
            ("toolz", "toolz==1.1.0"),
            ("numba", "numba==0.60.0"),
            ("skan", "skan==0.13.1 --no-deps"),
            ("networkx", "networkx==3.2.1"),
            ("pandas", "pandas==2.3.3"),
            ("torch", "torch==2.2.2"),
            ("torch_geometric", "torch_geometric==2.6.1"),
            ("numpy", "numpy==1.26.4"),
        ]

        missing = []
        for import_name, pip_name in packages:
            try:
                __import__(import_name)
            except ImportError:
                missing.append((import_name, pip_name))

        if missing:
            ret = qt.QMessageBox.question(
                None,
                "NeuroCTA: Install dependencies",
                "The following packages need to be installed:\n\n- " + "\n- ".join(name for name, _ in missing) + "\n\nThis may take a few minutes. Proceed?"
            )
            if ret == qt.QMessageBox.No:
                print("NeuroCTA: Dependency install cancelled by user.")
                return

            for import_name, pip_name in missing:
                try: 
                    slicer.util.showStatusMessage(f"NeuroCTA: Installing {import_name}...")
                    slicer.app.processEvents()
                    slicer.util.pip_install(pip_name)
                except Exception as e:
                    print(f"NeuroCTA: Failed to install {pip_name}: {e}")

        # Always ensure numpy<2
        try:
            import numpy as np
            if tuple(int(x) for x in np.__version__.split(".")[:2]) >= (2, 0):
                slicer.util.showStatusMessage("NeuroCTA: Downgrading numpy...")
                slicer.app.processEvents()
                slicer.util.pip_install("numpy<2")
        except Exception as e:
            print(f"NeuroCTA: Failed to downgrade numpy: {e}")

#
# NeuroCTALogic
#


class NeuroCTALogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

#
# NeuroCTATest
#


class NeuroCTATest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_NeuroCTA1()

    def test_NeuroCTA1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        pass