import PIL.Image
import os, time
import numpy as np, tifffile, torch
from skimage import filters, morphology
SRC = r'S:\micro\arb\hp2655\20260818_PNX_IMARE-148212\smc\concatenated'
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True).cuda().eval()

def well_mask(p):
    sm = filters.gaussian(p, sigma=20, preserve_range=True)
    w = sm > 0.5 * np.percentile(sm, 75)
    w = morphology.remove_small_objects(w, 10000)
    w = morphology.remove_small_holes(w, 100000)
    return morphology.binary_erosion(w, morphology.disk(15))

for wid in ['I10', 'H08']:
    if os.path.exists(f'output/feats_fr/{wid}_t15.npz'): print(wid,'cached'); continue
    tl = tifffile.imread(f'{SRC}\{wid}_montage.tif')[..., 2].astype(np.float32)
    t0 = time.time()
    for t in range(tl.shape[0]):
        w = well_mask(tl[t])
        med = np.median(tl[t][w])
        bg = filters.gaussian(np.where(w, tl[t], med), sigma=80, preserve_range=True)
        norm = np.where(w, tl[t]/(bg+1e-6), 1.0).astype(np.float32)
        xp = np.zeros((3248,3248), np.float32); xp[:3240,:3240] = np.clip((norm-0.5)/0.8, 0, 1)
        xt = ((torch.from_numpy(xp)[None,None].repeat(1,3,1,1) - IMAGENET_MEAN) / IMAGENET_STD).cuda()
        with torch.no_grad(), torch.autocast('cuda', torch.float16):
            pt = model.forward_features(xt)['x_norm_patchtokens'][0].reshape(232,232,768).half().cpu().numpy()
        np.savez(f'output/feats_fr/{wid}_t{t}.npz', feats=pt, norm=norm.astype(np.float16), well=w)
    print(f'{wid}: {time.time()-t0:.0f}s', flush=True)
print('DONE')
