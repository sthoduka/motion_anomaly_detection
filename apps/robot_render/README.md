Generates rendered images of the robot body from the point of view of its camera using the robot's URDF (and corresponding 3D meshes), and the joint positions.

<p align="center">
    <img src="https://raw.githubusercontent.com/sthoduka/motion_anomaly_detection/main/apps/robot_render/samples/sample1.jpg" height=120/>
    <img src="https://raw.githubusercontent.com/sthoduka/motion_anomaly_detection/main/apps/robot_render/samples/sample_sequence.gif" height=120/>
</p>

### Requirements

* urdfpy (for now use [this fork](https://github.com/sthoduka/urdfpy) since the original repository has not yet merged some fixes)
* [pyrender](https://github.com/mmatl/pyrender)
* PIL
* yaml
* numpy

### Resources

#### URDF
You will need the URDF of the robot you want to render. For the HSR robot, which was used to generate [this dataset](https://zenodo.org/record/4578539), the URDF files can be found [here](https://github.com/ToyotaResearchInstitute/hsr_description).

Clone the repositories [hsr_description](https://github.com/ToyotaResearchInstitute/hsr_description) and [hsr_meshes](https://github.com/ToyotaResearchInstitute/hsr_meshes) into your ros workspace and build them with catkin.

Note: There is currently a typo in the URDF file for `hsrb4s`; a [PR](https://github.com/ToyotaResearchInstitute/hsr_description/pull/9) to fix this has not been merged yet. You can apply the following patch:

```
diff --git a/robots/hsrb4s.urdf b/robots/hsrb4s.urdf
index 029b2e8..8051d3f 100644
--- a/robots/hsrb4s.urdf
+++ b/robots/hsrb4s.urdf
@@ -461,7 +461,7 @@ POSSIBILITY OF SUCH DAMAGE.
     </visual>
     <collision>
       <geometry>
-        <mesh filename="package://hsr_meshes/meshes/torso_v0/torso.std"/>
+        <mesh filename="package://hsr_meshes/meshes/torso_v0/torso.stl"/>
       </geometry>
     </collision>
   </link>
diff --git a/urdf/torso_v0/torso.urdf.xacro b/urdf/torso_v0/torso.urdf.xacro
index 8f83b9c..148cd4b 100644
--- a/urdf/torso_v0/torso.urdf.xacro
+++ b/urdf/torso_v0/torso.urdf.xacro
@@ -64,7 +64,7 @@ POSSIBILITY OF SUCH DAMAGE.
             <collision>
                 <geometry>
                     <xacro:unless value="${g_use_obj_for_collision}">
-                        <mesh filename="package://hsr_meshes/meshes/torso_v0/torso.std" />
+                        <mesh filename="package://hsr_meshes/meshes/torso_v0/torso.stl" />
                     </xacro:unless>
                     <xacro:if value="${g_use_obj_for_collision}">
                         <mesh filename="package://hsr_meshes/meshes/torso_v0/torso.collision.obj" />
```


#### Intrinsic camera matrix
You need the intrinsic camera matrix of the camera of your robot. The provided `rgb_camera_calibration.txt` contains the calibration parameters of the Asus Xtion Pro on the Toyota HSR which was used to collect our dataset.

### Run
```
python3 render_robot_body.py <path to trial folder> <path to URDF file>
```
Example:
```
python3 render_robot_body.py /media/ubuntu/data/validation/trial_115 /home/ubuntu/catkin_ws/src/hsr_description/robots/hsrb4s.urdf
```

If you want to preview the rendered images without saving them, append `-p` to the above command

