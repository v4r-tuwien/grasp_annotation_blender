import bpy
import numpy as np
import os
import mathutils
from math import radians
import math

"""
Author: Philipp Feigl
Adapted by: Daniel Zimmer
"""

###############################################################
###############################################################
###############################################################

# Change this to your object (must only be substring of name) #
obj_substring = "sugar"

# Change your gripper (True = HSR, False = PAL) 
use_hsr = False

###############################################################
###############################################################
###############################################################

if use_hsr is True:
    hand_name = 'hsr_hand'
    armature_name = 'hsr_armature'
    bone_name = "Left"
    hand_offset = 0.05
else:
    hand_name = 'pal_gripper'
    armature_name = "pal_armature_02"
    bone_name = "Bone"
    hand_offset = 0.05

opposite = True

cwd = bpy.path.abspath('//')
annotations_path = os.path.join(cwd, "annotations")
models_path = os.path.join(cwd, "models")

obj_arrays = ['003_cracker_box',
              '004_sugar_box',
              '006_mustard_bottle',
              '011_banana',
              '013_apple',
              '014_lemon',
              '015_peach',
              '016_pear',
              '017_orange',
              '018_plum',
              '024_bowl',
              'fluidcontainer',
              'largerinsefluidabottle', 
              'smallsoybrothbottle']
              
obj_name = None
for obj in obj_arrays:
    if obj_substring in obj:
        obj_name = obj
        break
    
if not obj_name:
    raise Exception(f"Object not found. Is the path correct?") 

    
npy_file_name = os.path.join(annotations_path, obj_name + '.npy')
stl_file_name = os.path.join(models_path, obj_name + '.stl')
bpy.ops.import_mesh.stl(filepath=stl_file_name)

stl_object = bpy.data.objects.get(obj_name)

if stl_object:
    stl_object.scale = (0.001, 0.001, 0.001)
else:
    print("STL object not found in the scene.")
    
###############################################################
# Grasp Annotation for top and side grasps
###############################################################

class OBJECT_PT_GraspAnnotationPanel(bpy.types.Panel):
    """
    Creates a Panel in the Object properties window to move gripper to a top or side grasp
    """
    bl_label = "Grasp Annotation"
    bl_idname = "PT_GraspAnnotationPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TFs'

    def draw(self, context):
        """
        Draws the panel
        
        Functions: 
            Operator:
                Top Grasp (object.annotate_grasp): Moves the gripper to a top grasp
                Side Grasp (object.annotate_grasp): Moves the gripper to a side grasp
        """
        layout = self.layout

        hand_obj = bpy.data.objects.get(hand_name)
        obj = bpy.data.objects.get(obj_name)
        if obj:
            layout.label(text="Grasp Annotation:")
            row = layout.row(align=True)
            row.operator("object.annotate_grasp", text="Top Grasp").grasp_type = 'TOP'
            row.operator("object.annotate_grasp", text="Side Grasp").grasp_type = 'SIDE'

        else:
            layout.label(text="Select an Armature object for annotation.")

"""
Calculate vertices and bb orientation
"""
def calculate_bb_rot(obj, bb_center, label, opposite=opposite):
    world_matrix = obj.matrix_world
    if label=='xy':
        vertices = [world_matrix @ v.co for v in obj.data.vertices]
        filtered_vertices = list(filter(lambda v: abs(v.z - bb_center.z) < 0.005, vertices))
        # sorted_vertices = sorted(filtered_vertices, key=lambda v: (v.x - bb_center.x)**2 + (v.y - bb_center.y)**2, reverse=True)
        sorted_length_vertices = sorted(filtered_vertices, key=lambda v: v.length, reverse=True)
        # far_vertices = sorted_length_vertices[0:4]
        # print(f"{far_vertices=}")
        farthest_vertices = sorted_length_vertices[0]
        print(f"{farthest_vertices=}")
        if not opposite:
            vertices_xy = farthest_vertices
        else:
            opp_vertices = sorted(vertices, key=lambda v: (v.x + farthest_vertices.x)**2 + (v.y + farthest_vertices.y)**2 + (v.z - farthest_vertices.z)**2, reverse=False)
            vertices_xy = opp_vertices[0]
        
        print(f"{vertices_xy=}")
        direction_to_vertices = (vertices_xy - bb_center).to_2d()
        print(f"{direction_to_vertices=}")
        direction_to_vertices = (vertices_xy - bb_center).normalized().to_2d()
        print(f"{direction_to_vertices=}")
        #rotation_quaternion = direction_to_vertices.to_track_quat('Z', 'Y')
        x_axis_vector = mathutils.Vector((1, 0))
        angle_z = math.degrees(direction_to_vertices.angle_signed(x_axis_vector)) * np.pi/180
        print(f"{angle_z * 180/np.pi=}")
        return angle_z, vertices_xy
        

class OBJECT_OT_AnnotateGraspOperator(bpy.types.Operator):
    bl_idname = "object.annotate_grasp"
    bl_label = "Annotate Grasp"
    bl_options = {'REGISTER', 'UNDO'}

    grasp_type: bpy.props.StringProperty(default='TOP')

    def execute(self, context):
        hand = bpy.data.objects.get(hand_name)
        obj = bpy.data.objects.get(obj_name)
        if obj and hand:
            bb_center, bb_rotation = get_bb_pose(obj)
            angle_z, vertices_xy = calculate_bb_rot(obj, bb_center, 'xy')
            bb_dimensions = obj.dimensions
            if self.grasp_type == 'TOP':
                print("Annotating Top Grasp")
                hand.location.x = bb_center.x
                hand.location.y = bb_center.y
                hand.location.z = bb_center.z + bb_dimensions[2]/2 + hand_offset
                hand.rotation_euler.x = 0
                hand.rotation_euler.y = np.pi
                hand.rotation_euler.z = angle_z
                print(f"{bb_rotation=}")
            elif self.grasp_type == 'SIDE':
                r, theta, phi = cartesian_to_spherical(vertices_xy[0], vertices_xy[1], vertices_xy[2])
                r_new = r + hand_offset/2
                x_new, y_new, z_new = spherical_to_cartesian(r_new, theta, phi)
                print("Annotating Side Grasp")
                hand.location.x = x_new
                hand.location.y = y_new
                hand.location.z = z_new
                # hand.rotation_euler.x = np.pi/2 - bb_euler.x
                hand.rotation_euler.x = 0
                hand.rotation_euler.y = -np.pi/2
                hand.rotation_euler.z = angle_z
        return {'FINISHED'}

###############################################################
# Rotate around bounding box center
###############################################################

class OBJECT_PT_RotateAroundPointPanel(bpy.types.Panel):
    """
    Creates a Panel in the Object properties window to rotate the gripper around the bounding box center
    """
    bl_label = "Rotate around bb center (z-axis)"
    bl_idname = "PT_RotateAroundPointPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TFs'

    def draw(self, context):
        """
        Draws the panel
        
        Functions:
            FloatProperty: 
                Rotation Angle (context.scene.rotation_angle): Angle to rotate the object around the bounding box center
            Operator:
                Rotate Object (object.rotate_around_point): Rotates the object around the bounding box center

        """
        layout = self.layout
    
        row = layout.row()
        row.label(text="Set rotation angle around z-axis (degrees):")
        row = layout.row()
        row.prop(context.scene, "rotation_angle", text="Rotation Angle")

        layout.operator("object.rotate_around_point", text="Rotate Object")

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))
    phi = np.arctan2(y, x)
    theta = np.arccos(z/r)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x, y, z

def get_bb_pose(obj):
    world_bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    bbox_center = sum(world_bbox, mathutils.Vector()) / 8
    rotation_matrix = mathutils.Matrix.Translation(bbox_center).to_3x3().inverted() @ obj.matrix_world.to_3x3()
    return bbox_center, rotation_matrix

def rotate_around_point(hand_obj, rotation_angle):
    data = bpy.data
    i = 0
    for obj in data.objects:
        if obj.name != hand_name:
            bb_center, rotation_matrix = get_bb_pose(obj)
            break
    # Calculate vector from bb center to current position
    vector_to_rotation_point = hand_obj.location - mathutils.Vector(bb_center)
    x, y, z = np.asarray(vector_to_rotation_point, dtype=np.float32)
    r, theta, phi = cartesian_to_spherical(x, y, z)
    phi += rotation_angle * np.pi/180
    x_new, y_new, z_new = spherical_to_cartesian(r, theta, phi)
    hand_obj.location.x = bb_center.x + x_new
    hand_obj.location.y = bb_center.y + y_new
    hand_obj.location.z = bb_center.z + z_new
    hand_obj.rotation_euler.z += rotation_angle * np.pi/180

def rotate_around_point_operator(rotation_angle):
    hand_obj = bpy.data.objects.get(hand_name)
    if hand_obj:
        rotate_around_point(hand_obj, rotation_angle)

class OBJECT_OT_RotateAroundPointOperator(bpy.types.Operator):
    bl_idname = "object.rotate_around_point"
    bl_label = "Rotate Object Around Point"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rotate_around_point_operator(context.scene.rotation_angle)
        return {'FINISHED'}

###############################################################
# Translation and Rotation Control
###############################################################

class OBJECT_PT_TranslationRotationPanel(bpy.types.Panel):
    """
    Creates a Panel in the Object properties window to translate and rotate the gripper
    """
    bl_label = "Translation and Rotation Control"
    bl_idname = "PT_TranslationRotationPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TFs'

    def draw(self, context):
        """
        Draws the panel

        Functions:
            FloatProperty:
                Translation Step (context.scene.translation_step): Step size for translation in meter
                Location (obj.location): Location of the object (x, y, z)
                Rotation Step (context.scene.rotation_step): Step size for rotation in degrees
                Rotation (obj.rotation_euler): Rotation of the object (x, y, z)
            Operator:
                X+, X-, Y+, Y-, Z+, Z- (object.translate): Translates the object in x, y, or z direction respectively. The distance is equal to the translation step
                X+, X-, Y+, Y-, Z+, Z- (object.rotate): Rotates the object around x, y, or z axis respectively. The angle is equal to the rotation step
        """
        layout = self.layout

        obj = bpy.data.objects.get(hand_name)

        if obj:

            # Translation-Buttons
            layout.prop(context.scene, "translation_step", text="Translation Step")
            row = layout.row(align=True)
            # layout.prop(context.scene, "obj.location.x", text="X: {:.4f}".format(obj.location.x))
            layout.prop(obj, "location", slider=True)
            # layout.label(text="X: {:.4f}".format(obj.location.x))
            row.operator("object.translate", text=f"X+").direction = 'X+'
            row.operator("object.translate", text=f"X-").direction = 'X-'
            # layout.label(text="Y: {:.4f}".format(obj.location.y))
            row.operator("object.translate", text=f"Y+").direction = 'Y+'
            row.operator("object.translate", text=f"Y-").direction = 'Y-'
            # layout.label(text="Z: {:.4f}".format(obj.location.z))
            row.operator("object.translate", text=f"Z+").direction = 'Z+'
            row.operator("object.translate", text=f"Z-").direction = 'Z-'

            # Rotation-Buttons
            layout.prop(context.scene, "rotation_step", text="Rotation Step")
            row = layout.row(align=True)
            layout.prop(obj, "rotation_euler", text="Rot", slider=True)
            row.operator("object.rotate", text="X+").axis = 'X+'
            row.operator("object.rotate", text="X-").axis = 'X-'
            row.operator("object.rotate", text="Y+").axis = 'Y+'
            row.operator("object.rotate", text="Y-").axis = 'Y-'
            row.operator("object.rotate", text="Z+").axis = 'Z+'
            row.operator("object.rotate", text="Z-").axis = 'Z-'


        else:
            layout.label(text=f"Object " + hand_name + " not found.")


class OBJECT_OT_TranslateOperator(bpy.types.Operator):
    bl_idname = "object.translate"
    bl_label = "Translate Object"
    bl_options = {'REGISTER', 'UNDO'}

    direction: bpy.props.StringProperty(default='X')

    def execute(self, context):
        obj = bpy.data.objects.get(hand_name)
        if obj:
            translation_vector = [0, 0, 0]
            if self.direction == 'X+':
                translation_vector[0] = context.scene.translation_step
            elif self.direction == 'X-':
                translation_vector[0] = -context.scene.translation_step
            elif self.direction == 'Y+':
                translation_vector[1] = context.scene.translation_step
            elif self.direction == 'Y-':
                translation_vector[1] = -context.scene.translation_step
            elif self.direction == 'Z+':
                translation_vector[2] = context.scene.translation_step
            elif self.direction == 'Z-':
                translation_vector[2] = -context.scene.translation_step

            obj.location += mathutils.Vector(translation_vector)

        return {'FINISHED'}


class OBJECT_OT_RotateOperator(bpy.types.Operator):
    bl_idname = "object.rotate"
    bl_label = "Rotate Object"
    bl_options = {'REGISTER', 'UNDO'}

    axis: bpy.props.StringProperty(default='X')

    def execute(self, context):
        obj = bpy.data.objects.get(hand_name)
        if obj:
            rotation_euler = obj.rotation_euler

            if self.axis == 'X+':
                current_state = (rotation_euler.x*(180/np.pi) + context.scene.rotation_step) % 360
                rotation_euler.x = current_state * (np.pi/180)
            if self.axis == 'X-':
                current_state = (rotation_euler.x*(180/np.pi) - context.scene.rotation_step) % 360
                rotation_euler.x = current_state * (np.pi/180)
            elif self.axis == 'Y+':
                current_state = (rotation_euler.y*(180/np.pi) + context.scene.rotation_step) % 360
                rotation_euler.y = current_state * (np.pi/180)
            elif self.axis == 'Y-':
                current_state = (rotation_euler.y*(180/np.pi) - context.scene.rotation_step) % 360
                rotation_euler.y = current_state * (np.pi/180)
            elif self.axis == 'Z+':
                current_state = (rotation_euler.z*(180/np.pi) + context.scene.rotation_step) % 360
                rotation_euler.z = current_state * (np.pi/180)
            elif self.axis == 'Z-':
                current_state = (rotation_euler.z*(180/np.pi) - context.scene.rotation_step) % 360
                rotation_euler.z = current_state * (np.pi/180)

            obj.rotation_euler = rotation_euler

        return {'FINISHED'}

###############################################################
# Gripper Control
###############################################################

class OBJECT_PT_GripperControlPanel(bpy.types.Panel):
    """
    Creates a Panel in the Object properties window to open or close the gripper
    """
    bl_label = "Gripper Control"
    bl_idname = "PT_GripperControlPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TFs'

    def draw(self, context):
        """
        Draws the panel

        Functions:
            FloatProperty:
                Gripper Angle (context.scene.gripper_step): Slider to close the gripper (0-100%), updates automatically
        """
        
        layout = self.layout

        obj = bpy.data.objects.get(armature_name)
        
        if obj:
            layout.prop(context.scene, "gripper_step", text="Gripper Closing [%]")
        else:
            layout.label(text=f"Armature " + hand_name + " not found.")
            
class OBJECT_OT_GripperOperator(bpy.types.Operator):
    bl_idname = "object.gripper_move"
    bl_label = "Translate Object"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = bpy.data.objects.get(armature_name)
        if obj:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='POSE')
            
            bone = obj.pose.bones[bone_name]
            if bone:
                if use_hsr:
                    bone.rotation_euler = (0, 0, math.radians(context.scene.gripper_step / 2.5))
                else:
                    bone.location = mathutils.Vector([context.scene.gripper_step / 2380.95, 0, 0])
                
            bpy.ops.object.mode_set(mode='OBJECT') 

        return {'FINISHED'}

def gripper_step_update(self, context):
    bpy.ops.object.gripper_move()
    
###############################################################
# Move to grasping point
###############################################################

class OBJECT_PT_GPMovePanel(bpy.types.Panel):
    """
    Creates a Panel in the Object properties window move to a grasping point from the npy file
    """
    bl_label = "Grasping Point Mover"
    bl_idname = "PT_GPMovePanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TFs'

    def draw(self, context):
        """
        Draws the panel

        Functions:
            Operator:
                 Next (object.select_gp): Moves gripper the next grasp point from the npy file
                 Prev. (object.select_gp): Moves gripper the previous grasp point from the npy file
        """
        
        layout = self.layout

        obj = bpy.data.objects.get(hand_name)
        
        if obj:
            row = layout.row(align=True)
            row.operator("object.select_gp", text=f"Next").direction = '+'
            row.operator("object.select_gp", text=f"Prev.").direction = '-'
        else:
            layout.label(text=f"Armature " + hand_name + " not found.")
            
class OBJECT_OT_GPSelectOperator(bpy.types.Operator):
    bl_idname = "object.select_gp"
    bl_label = "Select Grasppoint"
    bl_options = {'REGISTER', 'UNDO'}

    direction: bpy.props.StringProperty(default='+')
    idx: bpy.props.IntProperty(default=0)
        
    def execute(self, context):
        obj = bpy.data.objects.get(hand_name)
        if obj:
            data = np.load(npy_file_name)
    
    
            print(self.idx)
            
            try:
                len_data = len(data)
            except:
                print("Couldn't move to position. There are no grasp points defined.")
                return {'CANCELLED'}

            if len_data < self.idx:
                self.idx = 0
            elif self.idx == len_data - 1 and self.direction == '+':
                self.idx = 0
            elif self.idx == 0 and self.direction == '-':
                self.idx = len_data - 1
            elif self.direction == '+':
                self.idx += 1
            elif self.direction == '-':
                self.idx -= 1
                
            matrix = data[self.idx][0].reshape((4, 4))
            rotation_matrix = mathutils.Matrix(matrix).to_3x3()
            translation_vector = matrix[:3, 3]
                
            obj.rotation_euler = rotation_matrix.to_euler()
            obj.location = translation_vector
            

        return {'FINISHED'}

def gripper_step_update(self, context):
    bpy.ops.object.gripper_move()

###############################################################
# Save Location
###############################################################

class SaveLocationPanel(bpy.types.Panel):
    """
    Creates a Panel in the Object properties window to save the pose of the gripper
    """
    bl_label = "Save Location"
    bl_idname = "PT_SaveLocationPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TFs'

    def draw(self, context):
        """
        Draws the panel

        Functions:
            Operator:
                Save Object Location (object.save_location): Saves the current location of the object
                Delete all (object.remove_locations): Deletes all saved locations
        """
        layout = self.layout

        row = layout.row()
        row.label(text=f'Obj: {obj_name}')
        row = layout.row()
        row = layout.row()
        row = layout.row()
        row.operator("object.save_location", text="Save tf", icon='CUBE')
        row.label(text=f'{get_number_of_entries()}')
        row = layout.row()
        row.operator("object.remove_locations", text="Delete all", icon='TRASH')

def get_number_of_entries():
    try:
        saved_locations = np.load(npy_file_name)
        return saved_locations.shape[0]
    except FileNotFoundError:
        return 0

def remove_location_callback(self, context):
        saved_locations = np.zeros((0, 1, 16))
        np.save(npy_file_name, saved_locations)

class OBJECT_OT_RemoveLocationsOperator(bpy.types.Operator):
    bl_idname = "object.remove_locations"
    bl_label = "Delete Object Locations"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        remove_location_callback(self, context)
        return {'FINISHED'}

def save_location_callback(self, context):
    # obj = context.active_object    hand = bpy.data.objects[hand_name]
    if hand is not None:
        hom_mat = hand.matrix_world
        flattened_hom_mat = np.asarray(hom_mat).flatten().reshape(1, 1, 16)
        save_location(flattened_hom_mat)

def save_location(location):
    try:
        saved_locations = np.load(npy_file_name)
    except FileNotFoundError:
        saved_locations = np.zeros((0, 1, 16))
    print(f"TEST")
    saved_locations = np.vstack([saved_locations, location])
    np.save(npy_file_name, saved_locations)
    
    print(f"{saved_locations=}")

class OBJECT_OT_SaveLocationOperator(bpy.types.Operator):
    bl_idname = "object.save_location"
    bl_label = "Save Object Location"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        save_location_callback(self, context)
        return {'FINISHED'}

###############################################################
# Register and Unregister
###############################################################

def register():
    bpy.utils.register_class(SaveLocationPanel)
    bpy.utils.register_class(OBJECT_OT_SaveLocationOperator)
    bpy.utils.register_class(OBJECT_OT_RemoveLocationsOperator)
    ####
    bpy.utils.register_class(OBJECT_PT_TranslationRotationPanel)
    bpy.utils.register_class(OBJECT_OT_TranslateOperator)
    bpy.utils.register_class(OBJECT_OT_RotateOperator)
    bpy.types.Scene.translation_step = bpy.props.FloatProperty(default=1.0, min=0, soft_max=0.5, precision=2)
    bpy.types.Scene.rotation_step = bpy.props.FloatProperty(default=5.0, min=-180.0, max=180.0, precision=1)
    ####
    bpy.utils.register_class(OBJECT_PT_RotateAroundPointPanel)
    bpy.utils.register_class(OBJECT_OT_RotateAroundPointOperator)
    bpy.types.Scene.rotation_angle = bpy.props.FloatProperty(default=0, min=-360, soft_max=360.0, precision=3)
    ####
    bpy.utils.register_class(OBJECT_PT_GripperControlPanel)
    bpy.utils.register_class(OBJECT_OT_GripperOperator)
    bpy.types.Scene.gripper_step = bpy.props.FloatProperty(default=0, min=0, max=100, precision=1, update=gripper_step_update)
    ####
    bpy.utils.register_class(OBJECT_PT_GPMovePanel)
    bpy.utils.register_class(OBJECT_OT_GPSelectOperator)
    ####
    bpy.utils.register_class(OBJECT_PT_GraspAnnotationPanel)
    bpy.utils.register_class(OBJECT_OT_AnnotateGraspOperator)

def unregister():
    bpy.utils.unregister_class(SaveLocationPanel)
    bpy.utils.unregister_class(OBJECT_OT_SaveLocationOperator)
    bpy.utils.unregister_class(OBJECT_OT_RemoveLocationsOperator)
    ####
    bpy.utils.unregister_class(OBJECT_PT_TranslationRotationPanel)
    bpy.utils.unregister_class(OBJECT_OT_TranslateOperator)
    bpy.utils.unregister_class(OBJECT_OT_RotateOperator)
    del bpy.types.Scene.translation_step
    del bpy.types.Scene.rotation_step
    ####
    bpy.utils.unregister_class(OBJECT_PT_RotateAroundPointPanel)
    bpy.utils.unregister_class(OBJECT_OT_RotateAroundPointOperator)
    del bpy.types.Scene.rotation_point
    del bpy.types.Scene.rotation_angle
    ####
    bpy.utils.unregister_class(OBJECT_PT_GripperControlPanel)
    bpy.utils.unregister_class(OBJECT_OT_GripperOperator)
    del bpy.types.Scene.gripper_step
    ####
    bpy.utils.unregister_class(OBJECT_PT_GPMovePanel)
    bpy.utils.unregister_class(OBJECT_OT_GPSelectOperator)
    ####
    bpy.utils.unregister_class(OBJECT_PT_GraspAnnotationPanel)
    bpy.utils.unregister_class(OBJECT_OT_AnnotateGraspOperator)

if __name__ == "__main__":
    register()