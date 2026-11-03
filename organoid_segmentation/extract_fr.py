"""Full-res DINOv2 features: 3240 -> pad 3248 -> 232x232 tokens (14px each)."""
import PIL.Image
import glob, os, time
import numpy as np, tifffile, torch
from skimage import filters, morphology

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True).cuda().eval()

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

for path in sorted(glob.glob('*_montage.tif')):
    wid = path.split('_')[0]
    if os.path.exists(f'output/feats_fr/{wid}_t15.npz'):
        print(wid, 'cached'); continue
    t0 = time.time()
    tl = tifffile.imread(path)[..., 2].astype(np.float32)
    for t in range(tl.shape[0]):
        w = well_mask(tl[t])
        norm = normalize(tl[t], w)
        x = np.clip((norm - 0.5) / 0.8, 0, 1)
        xp = np.zeros((3248, 3248), np.float32); xp[:3240, :3240] = x
        xt = torch.from_numpy(xp)[None, None].repeat(1, 3, 1, 1)
        xt = ((xt - IMAGENET_MEAN) / IMAGENET_STD).cuda()
        with torch.no_grad(), torch.autocast('cuda', torch.float16):
            out = model.forward_features(xt)
        pt = out['x_norm_patchtokens'][0].reshape(232, 232, 768).half().cpu().numpy()
        np.savez(f'output/feats_fr/{wid}_t{t}.npz',
                 feats=pt, norm=norm.astype(np.float16), well=w)
    print(f'{wid}: {time.time()-t0:.0f}s', flush=True)
print('DONE')
