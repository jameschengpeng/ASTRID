"""The data panels of the paper's figures, drawn from the outputs of ``reproduce/``.

Each panel is drawn with the same values, examples, windows, marks and colors as in the paper, by matplotlib
with its defaults, and saved on its own with a record of what it shows (``<panel>.json``). The schematic
panels (Fig. 1A and 1D, 2A, 3A, 5A), the paper's typography, panel sizes and page layout are not part of
this code. The example of each example panel is chosen by the rule given in its module's docstring and in
the Supplementary Section "Example panels".

- ``cai1``: Figure 4, D3 (``python -m astrid_repro.figures cai1``);
- ``public``: the supplementary figure of D2, D4 and D5 (``python -m astrid_repro.figures public``), with TUnCaT's
  evaluation (GPL-2, ``reproduce/tuncat_eval``) handed in at run time for the example panels;
- ``common``: the methods' names, colors and order, the marks of the tests, the plot of the scores.

The panels of D1, D6 and D7 (Figures 1-3, 5 and 6, Supplementary Figures S1-S3) are to come with the
release of those datasets.
"""
