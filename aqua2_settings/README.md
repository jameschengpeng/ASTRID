# AQuA2 settings used in the paper

ASTRID takes astrocyte territory as an input and does not include AQuA2
([yu-lab-vt/AQuA2](https://github.com/yu-lab-vt/AQuA2), MATLAB, GPL-2). The paper's territory was made with
AQuA2 at commit `6e9a965`, with these settings files, in AQuA2's batch format (one column per preset):

| File | Data | Main settings |
|---|---|---|
| `D1.csv` | D1 (two-photon cortex) | threshold scale 5, minimum duration 15 frames (4.9 s at 3.09 Hz), minimum size 50 px, frame interval 0.324 s, 2 µm per pixel |
| `D6_default.csv` | D6 (the astrocyte channel of the two-color recordings) | AQuA2's defaults (threshold scale 3, minimum duration 5 frames, minimum size 20 px) |
| `D7_location1.csv` | D7, location 1 (one-photon spinal cord) | threshold scale 4, minimum duration 16 frames, minimum size 47 px, maximum size 706 px, frame interval 0.308 s, 2.07 µm per pixel |
| `D7_location2.csv` | D7, location 2 | as location 1, threshold scale 8 |

AQuA2 ran on a copy of each movie averaged in time to about 3 frames per second (D1: tenfold, 30.9 to 3.09 Hz),
on the same pixel grid as the movie ASTRID fits. The frame interval (`frameRate`, seconds per frame) and the
pixel size (`spatialRes`, µm) in each file describe that copy: change them for other data. AQuA2 drops
`regMaskGap` (5) pixels at each edge; `astrid.aqua2.territory` puts the events back on the full frame.

In MATLAB, with the AQuA2 folder as the working directory:

```matlab
aqua_cmd_batch('batch.csv', 'path/to/D1.csv')   % batch.csv: inputPath (and optional outputPath) per movie
```

Then, in Python, the territory of an imaging location is the pixels touched by at least two events over its
recordings:

```python
import astrid
territory = astrid.aqua2.territory(["rec1_AQuA2.mat", "rec2_AQuA2.mat"])      # (H, W) boolean
```
