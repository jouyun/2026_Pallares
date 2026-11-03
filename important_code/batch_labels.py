"""Batch: apply full-res DINOv2 + trained head to every *_montage.tif in a folder.
Writes <well>_organoid_labels.tiff next to each input. Resume-safe."""
import PIL.Image
import glob, os, sys, time, traceback
import numpy as np, tifffile, torch, joblib
from skimage import filters, morphology, measure, transform

SRC = r'S:\micro\arb\hp2655\20260818_PNX_IMARE-148212\smc\concatenated'
PROB_THR = 0.5
MIN_AREA = 3000

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True).cuda().eval()
clf = joblib.load('output/organoid_head_fr.joblib')  # v2: + mesh/glare negatives from I10, H08
W_COEF = torch.from_numpy(clf.coef_[0].astype(np.float32)).cuda()
B_COEF = float(clf.intercept_[0])
os.makedirs('output/qc_batch', exist_ok=True)

def well_mask(p):
    sm = filters.gaussian(p, sigma=20, preserve_range=True)
    w = sm > 0.5 * np.percentile(sm, 75)
    w = morphology.remove_small_objects(w, 10000)
    w = morphology.remove_small_holes(w, 100000)
    return morphology.binary_erosion(w, morphology.disk(15))

def normalize(p, w):
    med = np.median(p[w])
    bg = filters.gaussian(np.where(w, p, med), sigma=80, preserve_range=True)
    return np.where(w, p / (bg + 1e-6), 1.0).astype(np.float32)

def plane_mask(p):
    H, W = p.shape
    w = well_mask(p)
    if w.sum() < 50000:                     # empty/failed well image
        return np.zeros((H, W), bool)
    norm = normalize(p, w)
    Hp = (H + 13) // 14 * 14; Wp = (W + 13) // 14 * 14
    xp = np.zeros((Hp, Wp), np.float32)
    xp[:H, :W] = np.clip((norm - 0.5) / 0.8, 0, 1)
    xt = torch.from_numpy(xp)[None, None].repeat(1, 3, 1, 1)
    xt = ((xt - IMAGENET_MEAN) / IMAGENET_STD).cuda()
    with torch.no_grad(), torch.autocast('cuda', torch.float16):
        pt = model.forward_features(xt)['x_norm_patchtokens'][0]      # (N,768)
        pt = pt.float()
        pt = pt / (pt.norm(dim=1, keepdim=True) + 1e-8)
        logit = pt @ W_COEF + B_COEF
        prob = torch.sigmoid(logit).reshape(Hp//14, Wp//14).cpu().numpy()
    pfull = transform.resize(prob, (Hp, Wp), order=1)[:H, :W]
    m = (pfull > PROB_THR) & w
    m = morphology.remove_small_objects(m, MIN_AREA)
    m = morphology.remove_small_holes(m, 5000)
    return m

files = sorted(glob.glob(os.path.join(SRC, '*_montage.tif')))
only = sys.argv[1] if len(sys.argv) > 1 else None
print(f'{len(files)} montages found', flush=True)
done = errs = 0
from concurrent.futures import ThreadPoolExecutor
_pool = ThreadPoolExecutor(1)
def _read(p):
    try: return tifffile.imread(p)
    except Exception as e: return e

todo = []
for path in files:
    stem0 = os.path.basename(path)[:-len('.tif')]
    well0 = stem0[:-len('_montage')] if stem0.endswith('_montage') else stem0
    if only and well0 != only: continue
    if os.path.exists(os.path.join(SRC, f'{well0}_organoid_labels.tiff')): continue
    todo.append(path)
print(f'{len(todo)} to process', flush=True)
fut = _pool.submit(_read, todo[0]) if todo else None
for i, path in enumerate(todo):
    stem = os.path.basename(path)[:-len('.tif')]
    well = stem[:-len('_montage')] if stem.endswith('_montage') else stem
    outp = os.path.join(SRC, f'{well}_organoid_labels.tiff')
    t0 = time.time()
    try:
        img = fut.result()
        fut = _pool.submit(_read, todo[i+1]) if i+1 < len(todo) else None
        if isinstance(img, Exception): raise img
        tl = img[..., 2].astype(np.float32) if img.ndim == 4 else img.astype(np.float32)
        T = tl.shape[0]
        labs = np.zeros(tl.shape, np.uint16)
        nobj = []
        for t in range(T):
            m = plane_mask(tl[t])
            lab = measure.label(m)
            nobj.append(int(lab.max()))
            labs[t] = lab.astype(np.uint16)
        tmp = outp + '.part'
        tifffile.imwrite(tmp, labs, compression='zstd', photometric='minisblack')
        os.replace(tmp, outp)
        # small local QC thumb (t = middle plane)
        tm = T // 2
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        s = tl[tm][::4, ::4]; ms = labs[tm][::4, ::4] > 0
        lo, hi = np.percentile(s, [1, 99.5])
        rgb = np.clip((s-lo)/(hi-lo), 0, 1)[..., None].repeat(3, -1)
        rgb[ms] = rgb[ms]*0.6 + np.array([0.4, 0, 0])
        plt.imsave(f'output/qc_batch/{well}.jpg', np.clip(rgb, 0, 1))
        done += 1
        print(f'{well}: {time.time()-t0:.0f}s objs(t{tm})={nobj[tm]} [{done} done]', flush=True)
    except Exception:
        errs += 1
        print(f'ERROR {well}:\n{traceback.format_exc()}', flush=True)
print(f'FINISHED done={done} errors={errs}', flush=True)
