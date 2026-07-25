# nuScenes benchmark tasks, with examples

Reference notes on the five official nuScenes tasks. All tasks operate on the same unit: a
*keyframe* ("sample"), annotated at 2 Hz and identified by a `sample_token`.

Sources: [nuscenes.org](https://www.nuscenes.org/nuscenes),
[nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) eval READMEs,
[arXiv:1903.11027](https://arxiv.org/abs/1903.11027).

## Dataset in one paragraph

1,000 scenes of 20 s each (~5.5 h), driven in Boston and Singapore. Full 360° rig: 6 cameras,
1 32-beam lidar (20 Hz), 5 radars, GPS/IMU. ~1.4M images, ~390k lidar sweeps, ~1.4M 3D boxes over
23 classes and 8 attributes. Annotations land on 2 Hz keyframes; intermediate readings ("sweeps")
ship unlabeled. Licensed CC BY-NC-SA 4.0 — non-commercial only.

## 1. 3D object detection

One frame in, a list of 3D boxes out.

```json
{"sample_token": "ca9a282c9e77460f8360f564131a8af5",
 "translation": [412.5, 1130.9, 0.85],
 "size":        [1.90, 4.62, 1.73],
 "rotation":    [0.707, 0.0, 0.0, 0.707],
 "velocity":    [2.3, -0.1],
 "detection_name": "car",
 "detection_score": 0.91,
 "attribute_name": "vehicle.moving"}
```

- `translation` is the box center x, y, z in global metres; `size` is w, l, h; `rotation` is a
  quaternion (w, x, y, z); `velocity` is vx, vy in m/s.
- 10 classes: `car, truck, bus, trailer, construction_vehicle, pedestrian, motorcycle, bicycle,
  traffic_cone, barrier`.
- Metric: **NDS** = (5 · mAP + sum of the 5 TP scores) / 10. mAP matches by 2D center distance on
  the ground plane at thresholds {0.5, 1, 2, 4} m rather than IoU. The 5 TP scores come from
  translation, scale, orientation, velocity, and attribute errors, each mapped as
  `max(1 - error, 0)`.

## 2. Multi-object tracking

Detection output plus an identity that must persist across frames.

```json
// frame t
{"sample_token": "ca9a282c...", "tracking_id": "track_017",
 "tracking_name": "car", "tracking_score": 0.91,
 "translation": [412.5, 1130.9, 0.85], "size": [1.90, 4.62, 1.73], ...}

// frame t + 0.5 s — same car, same id
{"sample_token": "39586f9d...", "tracking_id": "track_017",
 "tracking_name": "car", "tracking_score": 0.88,
 "translation": [413.6, 1130.8, 0.85], "size": [1.90, 4.62, 1.73], ...}
```

- 7 classes: the detection list minus `barrier`, `traffic_cone`, `construction_vehicle`.
- `tracking_name` may not change within a track. Class-specific eval range: 40 m for
  bicycle/motorcycle, 50 m for the rest.
- Metric: **AMOTA** (primary) and **AMOTP** — MOTA/MOTP averaged over 40 recall thresholds.
  Reusing `track_017` for a different car is an ID switch and is penalized directly.

## 3. Motion prediction

Pick one agent in one frame; forecast where it goes.

- Input: an `(instance_token, sample_token)` pair, plus up to 2 s of that agent's past and the HD map.
- Output: up to 25 candidate futures, each 12 timesteps (6 s at 2 Hz) of global x-y, with a
  probability per mode.

```python
Prediction(instance="bc38961ca0ac...",
           sample="39586f9d5956...",
           prediction=np.zeros((25, 12, 2)),   # modes x timesteps x (x, y)
           probabilities=np.ones(25) / 25)
```

- Metrics: **minADE_k**, **minFDE_k**, **MissRate_2_k** (a miss is max pointwise L2 error > 2 m).
- Only the best of the k modes counts, so hedging across genuinely distinct futures — turn left
  vs. continue straight — is rewarded over averaging them into one blurred trajectory.

## 4. Lidar semantic segmentation

Label every point in the sweep. No boxes involved.

- Input: ~34k lidar points for the keyframe.
- Output: one `uint8` per point in the same order, written to
  `<lidar_sample_data_token>_lidarseg.bin`.

```
point 0 -> 24   # driveable_surface
point 1 -> 24
point 2 ->  4   # car
point 3 -> 30   # vegetation
```

- 16 evaluated classes: the 10 detection classes plus 6 stuff classes
  (`driveable_surface, other_flat, sidewalk, terrain, manmade, vegetation`). Index 0 is void and
  is excluded from scoring.
- Metric: **mIoU** (frequency-weighted IoU is reported but not used for ranking).
- Note this does not separate two adjacent cars — every car point is just "car".

## 5. Panoptic segmentation

Semantics *and* instances, packed into one per-point integer.

```
label = 1000 * challenge_class_index + per_category_instance_index
```

```
point 2 -> 4001    # class 4 (car), instance 1
point 7 -> 4002    # class 4 (car), instance 2  <- second car, now distinguished
point 0 -> 24000   # driveable_surface; stuff classes use instance 0
```

- Stored as `.npz`. Instance indices run 1..999 and reset per class within a scene.
- Same 16 challenge classes (10 thing + 6 stuff).
- Metrics: **PQ = SQ x RQ**. The panoptic *tracking* variant additionally requires instance ids to
  stay stable across the whole scene, scored by **PAT** and **LSTQ**.

## The progression

| Task | Adds over the previous |
| --- | --- |
| Detection | boxes in a frame |
| Tracking | boxes with memory |
| Prediction | boxes in the future |
| Lidar segmentation | per-point classes |
| Panoptic | per-point classes with identity |
