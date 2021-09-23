## Using Visual Anomaly Detection for Task Execution Monitoring
Santosh Thoduka, Juergen Gall and Paul G. Plöger

**Abstract**: Execution monitoring is essential for robots to detect and respond to failures. Since it is impossible to enumerate all failures for a given task, we learn from successful executions of the task to detect visual anomalies during runtime. Our method learns to predict the motions that occur during the nominal execution of a task, including camera and robot body motion. A probabilistic U-Net architecture is used to learn to predict optical flow, and the robot’s kinematics and 3D model are used to model camera and body motion. The errors between the observed and predicted motion are used to calculate an anomaly score. We evaluate our method on a dataset of a robot placing a book on a shelf, which includes anomalies such as falling books, camera occlusions, and robot disturbances. We find that modeling camera and body motion, in addition to the learning-based optical flow prediction, results in an improvement of the area under the receiver operating characteristic curve from 0.752 to 0.804, and the area under the precision-recall curve from 0.467 to 0.549.


## Supplementary video:
<iframe width="640" height="360" src="https://www.youtube.com/embed/U8dO8dEILZw" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Additional sample videos:

This video shows an example of an anomaly that occurs early, but there are no detections while the arm and book are static.
<iframe width="640" height="360" src="https://www.youtube.com/embed/-j0CMPbGfbM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

This video has no anomalies; intermittent false positives are detected during release and retraction.
<iframe width="640" height="360" src="https://www.youtube.com/embed/pym8hO7FsD8" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

This video has a large false positive because the body motion error goes slightly above its threshold. The anomaly at the start of the video is detected correctly though.
<iframe width="640" height="360" src="https://www.youtube.com/embed/FKi6l4pE6eM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
