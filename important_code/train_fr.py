"""Train head on FULL-RES tokens (14px). Interior-only positives; empty-space negatives inside pos boxes."""
import PIL.Image
import numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.ndimage import binary_erosion, binary_dilation

TOK = 14
G = 232

POS_BOXES = [
    ('A02', 13, 2700, 3060, 1550, 2500), ('A02', 13, 2740, 3060, 2700, 3100),
    ('A02', 13, 2760, 3060, 580, 900),
    ('E16', 0, 1680, 2160, 180, 1620),
    ('K22', 8, 1850, 2650, 650, 1450), ('K22', 8, 1790, 1920, 1590, 1730),
]
NEG_BOXES = [
    ('G16', 8, 350, 600, 2720, 2980), ('G16', 8, 2560, 2870, 560, 860),
    ('G16', 8, 1150, 2100, 2280, 2460), ('G16', 8, 1200, 2000, 600, 1600),
    ('A02', 13, 2150, 2700, 200, 1100), ('A02', 13, 430, 580, 600, 2600),
    ('A02', 13, 1300, 2000, 1300, 2100),
    ('E16', 0, 150, 1150, 300, 1400), ('E16', 0, 150, 1600, 1700, 2800),
    ('K22', 8, 500, 1500, 1900, 3050), ('K22', 8, 1000, 1750, 2000, 2800),
    ('K22', 8, 2200, 2800, 1700, 2700),
    ('K22', 8, 250, 400, 700, 2800), ('K22', 8, 400, 2700, 2950, 3100),
    ('K22', 8, 400, 1700, 200, 380), ('K22', 8, 2860, 2990, 700, 2600),
    ('G16', 8, 700, 2400, 200, 330), ('G16', 8, 260, 370, 700, 1400),
    ('G16', 8, 260, 370, 1900, 2600),
    ('A02', 13, 600, 2600, 3010, 3120),
    ('I10', 8, 2700, 2980, 1100, 1750),   # fibrous strand mesh
    ('I10', 8, 2450, 2980, 2150, 2400),   # fibrous strand mesh
    ('I10', 15, 2500, 2960, 1600, 2800),  # fibrous strand mesh + wall band
    ('I10', 15, 2400, 2750, 1200, 1550),  # strands above organoid
    ('H08', 8, 2500, 2980, 300, 640),     # corner + glare left of cluster
    ('H08', 8, 2600, 2980, 1700, 1900),   # glare/specks right of cluster
    ('H08', 8, 2500, 2600, 700, 1900),    # speck strip above cluster
    ('H08', 15, 2500, 2980, 1800, 2400),  # glare/specks right of cluster
    ('H08', 15, 2500, 2600, 800, 1750),   # speck strip above cluster
]
TRAIN_WELLS = ['A02', 'E16', 'K22', 'G16']

def load(wid, t):
    d = np.load(f'output/feats_fr/{wid}_t{t}.npz')
    return d['feats'].astype(np.float32), d['norm'].astype(np.float32), d['well']

def token_stats(norm):
    c = np.ones((G*TOK, G*TOK), np.float32); c[:3240, :3240] = norm
    v = c.reshape(G, TOK, G, TOK).transpose(0,2,1,3).reshape(G, G, TOK*TOK)
    return v.mean(-1), v.std(-1)

X, y = [], []
cache = {}
def get(wid, t):
    if (wid,t) not in cache: cache[(wid,t)] = load(wid,t)
    return cache[(wid,t)]

for wid, t, y0, y1, x0, x1 in POS_BOXES:
    f, norm, w = get(wid, t)
    tm, ts = token_stats(norm)
    box = np.zeros((G,G), bool); box[y0//TOK:y1//TOK, x0//TOK:x1//TOK] = True
    gate = (ts > 0.05) | (tm < 0.85)
    interior = binary_erosion(gate, np.ones((3,3)))       # token + all 8 neighbors textured
    pos = box & interior
    empty = box & ~binary_dilation(gate, np.ones((3,3)))  # clearly empty space inside pos box
    X.append(f[pos]); y.append(np.ones(pos.sum()))
    X.append(f[empty]); y.append(np.zeros(empty.sum()))

for wid, t, y0, y1, x0, x1 in NEG_BOXES:
    f, norm, w = get(wid, t)
    box = np.zeros((G,G), bool); box[y0//TOK:y1//TOK, x0//TOK:x1//TOK] = True
    X.append(f[box]); y.append(np.zeros(box.sum()))

# large verified organoid interiors (DINO-round labels), non-G16 wells
import tifffile
from skimage import measure, morphology as skm
for wid in ['A02', 'E16', 'K22']:
    labs = tifffile.imread(f'output/{wid}_organoid_labels.tif')
    for t in range(0, 16, 4):
        f, norm, w = get(wid, t)
        lab = labs[t]
        big = np.zeros_like(lab, bool)
        for r in measure.regionprops(lab):
            if r.area > 15000:
                big[r.slice][lab[r.slice] == r.label] = True
        big = skm.binary_erosion(big, skm.disk(25))
        c = np.zeros((G*TOK, G*TOK), bool); c[:3240, :3240] = big
        v = c.reshape(G, TOK, G, TOK).transpose(0,2,1,3).reshape(G, G, TOK*TOK)
        sel = v.all(-1)
        # also require organoid-texture gate so smooth halo areas from the coarse round don't leak in
        tm, ts = token_stats(norm)
        sel &= (ts > 0.05) | (tm < 0.85)
        X.append(f[sel]); y.append(np.ones(sel.sum()))

# generic smooth-background negatives
rng = np.random.default_rng(0)
for wid in TRAIN_WELLS:
    for t in [2, 10]:
        f, norm, w = get(wid, t)
        tm, ts = token_stats(norm)
        c = np.zeros((G*TOK, G*TOK), bool); c[:3240, :3240] = w
        v = c.reshape(G, TOK, G, TOK).transpose(0,2,1,3).reshape(G, G, TOK*TOK)
        wtok = v.all(-1)
        sel = wtok & (np.abs(tm-1) < 0.03) & (ts < 0.02)
        ii, jj = np.where(sel)
        k = rng.choice(len(ii), min(400, len(ii)), replace=False)
        X.append(f[ii[k], jj[k]]); y.append(np.zeros(len(k)))

X = np.concatenate(X); y = np.concatenate(y)
X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
print(f'tokens: {len(y)}  pos: {int(y.sum())}  neg: {int((1-y).sum())}')
clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000)
cv = cross_val_score(clf, X, y, cv=5, scoring='balanced_accuracy')
print('5-fold balanced acc: %.4f +/- %.4f' % (cv.mean(), cv.std()))
clf.fit(X, y)
joblib.dump(clf, 'output/organoid_head_fr.joblib')
print('saved output/organoid_head_fr.joblib')
