# The ROIs of D3 (CRCNS cai-1)

The CRCNS cai-1 dataset (Chen et al. 2013) ships no ROI masks. For the paper, the ROI of each of its 20 recorded
cells was drawn by hand once, as a polygon around the soma on the cell's mean image, blind to all methods. These
are those masks. Every method of the paper received them, through the export that `astrid_repro.cai1_export`
builds from the dataset.

Each file `<date>_cell<k>.npz` holds:

| Key | Content |
|---|---|
| `roi` | the mask, (256, 256) bool, on the full field of the raw movies |
| `polygon` | the vertices drawn, (m, 2), as (x, y) = (column, row) pixel coordinates |
| `source` | the recording whose mean image the polygon was drawn on |
| `drawer`, `date` | who drew it, and when |

Every recording of a cell uses the cell's mask (a file `<cell>_<rec>.npz` would give one recording its own mask;
there is none). The export checks each recording: on the unregistered movie, the mean of the mask must correlate
at least 0.95 with the dataset authors' own ROI trace (`fmean_roi`), so that the mask holds their cell. It
records the correlation as `roi_r` and bins the mask 4 × 4, keeping a binned pixel if at least half of its block
is in the mask.

The masks are the authors' annotation of the public dataset, under this repository's license (MIT). The dataset
itself is not included: it is available from CRCNS (https://crcns.org, cai-1).
