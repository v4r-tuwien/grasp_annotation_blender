# grasp_annotation_blender
Script to annotate grasp points in blender for the HSR.

The following panels are available in window TFs after script "LocationSave.py" is run in Blender:
(a) Save Location:
  Stores a numpy array of the flattened homogeneous tf matrix. The number of entries is displayed, as well as the objects name.
  It's possible to delete all entries.

(b) Translation and rotation control for the robot hand.

(c) Grasp annotations:
  Annotates grasp poses based on the object vertices. You can choose between TOP and SIDE grasp.

(d) Rotate around center of the bounding box of the object.

![grasp_annotations](https://github.com/philippfeigl/grasp_annotation_blender/assets/102654591/2d1e2709-bed5-4526-b35a-0125a58ddd46)
