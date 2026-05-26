# Dynamic Obstacle Detector

## Overall Pipeline

```text
LiDAR Scan
    ↓
Clustering
    ↓
Cluster Centroid Extraction
    ↓
Odometry Compensation
    ↓
Nearest-Neighbor Association
    ↓
Velocity Estimation
    ↓
Hit-count based Classification
```

---

## 1. LiDAR Clustering

Each LiDAR beam is converted into a 2D point:

$$
p_i = (r_i \cos \theta_i,\ r_i \sin \theta_i)
$$

Adjacent beams are grouped into the same cluster if:

$$
\|p_i - p_{i-1}\| < d_{jump}
$$

Current YAML setting:

```yaml
cluster_range_jump: 0.20
```

Meaning:

```text
If the distance between adjacent LiDAR beams is smaller than 20 cm,
they are considered part of the same obstacle.
```

A valid cluster must satisfy:

$$
|C_k| \ge 3
$$

$$
0.04 \le W_k \le 3.0
$$

where:

- \( |C_k| \): number of points in the cluster
- \( W_k \): cluster width

Current YAML settings:

```yaml
min_cluster_points: 3
min_cluster_width: 0.04
max_cluster_width: 3.0
```

---

## 2. Cluster Representation

Each obstacle is represented by its centroid:

$$
c_k^t =
\frac{1}{|C_k^t|}
\sum_{p_i \in C_k^t} p_i
$$

where:

- \( C_k^t \): cluster \( k \) at time \( t \)
- \( p_i \): LiDAR point inside the cluster
- \( c_k^t \): centroid of the cluster

The centroid is used as the representative obstacle position.

---

## 3. Odometry Compensation

Since the ego vehicle moves, even static obstacles appear to move in the LiDAR frame.

The previous centroid is transformed into the current LiDAR frame using odometry:

$$
\hat{c}_j^{t-1}
=
T_t^{-1} T_{t-1} c_j^{t-1}
$$

where:

- \( c_j^{t-1} \): previous centroid of track \( j \)
- \( \hat{c}_j^{t-1} \): previous centroid compensated into the current LiDAR frame
- \( T_{t-1} \): odometry transform at the previous frame
- \( T_t \): odometry transform at the current frame

Meaning:

```text
Odometry compensation is not velocity estimation itself.
It is a coordinate transformation that removes ego-motion.
```

After this transformation, the residual displacement between the compensated previous centroid and the current centroid is used to estimate obstacle motion.

---

## 4. Nearest-Neighbor Tracking

The detector associates the current cluster with the nearest compensated previous track.

Distance metric:

$$
d_{ij}
=
\left\|
c_i^t - \hat{c}_j^{t-1}
\right\|
$$

Nearest track:

$$
j^* = \arg\min_j d_{ij}
$$

Association condition:

$$
d_{ij^*} < 0.60m
$$

Current YAML setting:

```yaml
association_distance: 0.60
```

Meaning:

```text
If the compensated previous centroid and the current centroid
are within 60 cm, they are treated as the same obstacle.
```

---

## 5. Velocity Estimation

Obstacle velocity is estimated from centroid displacement after odometry compensation:

$$
v_i^t =
\frac{
\left\|
c_i^t - \hat{c}_i^{old}
\right\|
}{
t - t_{old}
}
$$

where:

- \( c_i^t \): current centroid
- \( \hat{c}_i^{old} \): old centroid transformed into the current LiDAR frame
- \( t - t_{old} \): time difference

Current YAML setting:

```yaml
track_history_size: 7
```

Meaning:

```text
Velocity is estimated using the displacement between the current centroid
and the odometry-compensated centroid history.
```

This is not a direct subtraction of velocity vectors.  
It is a displacement-based velocity estimate after ego-motion compensation.

---

## 6. Static / Dynamic Evidence Update

Static evidence:

$$
h_s^t =
\begin{cases}
h_s^{t-1} + 1, & v_i^t \le 0.6 \\
0, & otherwise
\end{cases}
$$

Dynamic evidence:

$$
h_d^t =
\begin{cases}
h_d^{t-1} + 1, & v_i^t \ge 0.8 \\
0, & otherwise
\end{cases}
$$

Current YAML settings:

```yaml
static_speed_threshold: 0.6
dynamic_speed_threshold: 0.8
```

Ambiguous region:

$$
0.6 < v_i^t < 0.8
$$

In this region:

```text
static_hit_count = 0
dynamic_hit_count = 0
```

and the previous label is maintained.

---

## 7. Motion Classification

Current YAML settings:

```yaml
min_observations: 5
static_confirm_count: 3
dynamic_confirm_count: 3
```

Final classification:

$$
L_i^t =
\begin{cases}
Unknown, & n_i^t < 5 \\
Static, & h_s^t \ge 3 \\
Dynamic, & h_d^t \ge 3 \\
L_i^{t-1}, & otherwise
\end{cases}
$$

where:

- \( L_i^t \): motion label of obstacle \( i \) at time \( t \)
- \( n_i^t \): observation count
- \( h_s^t \): static evidence count
- \( h_d^t \): dynamic evidence count

Meaning:

```text
The detector does not classify motion from a single frame.
Temporal hysteresis is used for robustness.
```

---

## 8. Density-based Static Reinforcement

Repeated observations at the same location reinforce static classification.

Current YAML settings:

```yaml
scan_density_history_size: 7
scan_density_cell_size: 0.08
scan_density_min_weight: 300
scan_density_cluster_min_dense_points: 2
```

Meaning:

```text
Clusters repeatedly observed in the same area
are strongly treated as static obstacles.
```

This helps reduce false dynamic classification caused by centroid fluctuation, LiDAR noise, or partial observations.

---

## 9. Wall-static Classification

Large static clusters are classified as wall-static.

Conditions:

$$
W_k \ge 1.0m
$$

$$
|C_k| \ge 10
$$

Current YAML settings:

```yaml
wall_static_min_width: 1.0
wall_static_min_points: 10
```

Meaning:

```text
Large structures such as walls or guardrails
are separated from ordinary static obstacles.
```

---

## Key Equations Summary

### Clustering

$$
\|p_i - p_{i-1}\| < 0.20
\Rightarrow same\ cluster
$$

### Centroid

$$
c_k =
\frac{1}{|C_k|}
\sum_{p_i \in C_k} p_i
$$

### Odometry Compensation

$$
\hat{c}^{t-1}
=
T_t^{-1}T_{t-1}c^{t-1}
$$

### Tracking Association

$$
j^* =
\arg\min_j
\left\|
c_i^t - \hat{c}_j^{t-1}
\right\|
$$

$$
d_{ij^*} < 0.60m
$$

### Velocity Estimation

$$
v =
\frac{
\left\|
c^t - \hat{c}^{old}
\right\|
}{
\Delta t
}
$$

### Classification

$$
v \le 0.6
\Rightarrow Static
$$

$$
v \ge 0.8
\Rightarrow Dynamic
$$

$$
0.6 < v < 0.8
\Rightarrow keep\ previous\ label
$$

---

## Summary

This dynamic obstacle detector uses LiDAR clustering and centroid-based tracking.  
The previous obstacle centroid is transformed into the current LiDAR frame using odometry compensation.  
Then, nearest-neighbor association connects current clusters with previous tracks.  
The residual centroid displacement is used to estimate obstacle velocity.  
Finally, static, dynamic, and unknown labels are determined using velocity thresholds, observation count, and hit-count based hysteresis.

---

## Dynamic PointCloud2 Output Fields

The `/dynamic_pointcloud` topic publishes only points that belong to clusters classified as dynamic in the latest input frame.

Each point contains the following fields:

| Field | Type | Description |
| --- | --- | --- |
| `x` | `float32` | Point x position in the current sensor frame. |
| `y` | `float32` | Point y position in the current sensor frame. |
| `z` | `float32` | Point z position. For LaserScan input, this is `0.0`. |
| `track_id` | `uint32` | Tracker ID of the dynamic cluster that contains this point. |
| `relative_speed` | `float32` | Odom-compensated speed of the dynamic cluster. |
| `relative_yaw` | `float32` | Bearing angle from the robot to the cluster center, computed as `atan2(center_y, center_x)`. |

Points from the same dynamic cluster share the same `track_id`, `relative_speed`, and `relative_yaw`.
