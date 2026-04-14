"""Render all v8 figures in one pass."""
import os
OUT = '/data/data/Drug_Pred/research/figures/figures_v8'
os.makedirs(OUT, exist_ok=True)
from . import fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8
for mod in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
    mod.make(OUT)
print('All v8 figures generated →', OUT)
