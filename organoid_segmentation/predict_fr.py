"""Full-res prediction: 232x232 prob -> bilinear -> threshold. No closing (boundaries stay tight)."""
import PIL.Image
import glob, os
import numpy as np, tifffile, joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import morphology, transform, measure

PROB_THR = 0.5
MIN_AREA = 3000

clf = joblib.load('output/organoid_head_fr.joblib')
os.makedirs('output/qc3', exist_ok=True)
wells = sorted({os.path.basename(f).split('_')[0] for f in glob.glob('output/feats_fr/*t0.npz')})
for wid in wells:
    bin_stack = np.zeros((16, 3240, 3240), np.uint8)
    for t in range(16):
        d = np.load(f'output/feats_fr/{wid}_t{t}.npz')
        f, w = d['feats'].astype(np.float32), d['well']
        ft = f.reshape(-1, 768)
        ft /= (np.linalg.norm(ft, axis=1, keepdims=True) + 1e-8)
        prob = clf.predict_proba(ft)[:, 1].reshape(232, 232)
        p = transform.resize(prob, (3248, 3248), order=1)[:3240, :3240]
        m = (p > PROB_THR) & w
        m = morphology.remove_small_objects(m, MIN_AREA)
        m = morphology.remove_small_holes(m, 5000)
        bin_stack[t] = np.where(m, 255, 0)
    tifffile.imwrite(f'output/{wid}_organoid_mask.tif', bin_stack,
                     compression='zstd', photometric='minisblack')
    labs = np.zeros(bin_stack.shape, np.uint16)
    for t in range(16):
        labs[t] = measure.label(bin_stack[t] > 0).astype(np.uint16)
    tifffile.imwrite(f'output/{wid}_organoid_labels.tif', labs, compression='zstd', photometric='minisblack')
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax, t in zip(axes, [0, 8, 15]):
        d = np.load(f'output/feats_fr/{wid}_t{t}.npz')
        norm, w = d['norm'].astype(np.float32), d['well']
        lo, hi = np.percentile(norm[w], [1, 99.5]) if w.any() else (0.7, 1.3)
        rgb = np.clip((norm[::2,::2] - lo) / (hi - lo), 0, 1)[..., None].repeat(3, -1)
        m = bin_stack[t][::2, ::2] > 0
        edge = morphology.binary_dilation(m, morphology.disk(2)) & ~m
        rgb[m] = rgb[m] * 0.65 + np.array([0.35, 0, 0]); rgb[edge] = [1, 0.1, 0.1]
        ax.imshow(rgb); ax.set_title(f'{wid} t={t}  area={100*m.mean():.1f}%'); ax.axis('off')
    plt.tight_layout(); plt.savefig(f'output/qc3/{wid}_qc.png', dpi=80); plt.close()
    print(wid, flush=True)
print('DONE')
