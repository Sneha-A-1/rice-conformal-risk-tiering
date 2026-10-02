"""Perceptual-hash near-duplicate audit (used for group-aware splitting)."""
import sys, json, pathlib, collections
import numpy as np, imagehash
from PIL import Image, ImageOps

def hashes(paths):
    out=[]
    for p in paths:
        im=Image.open(p).convert('L')
        hs=[]
        for k in range(4):
            r=im.rotate(90*k,expand=True)
            hs+=[imagehash.phash(r), imagehash.phash(ImageOps.mirror(r))]
        out.append(np.array([int(str(h),16) for h in hs],dtype=np.uint64))
    return np.stack(out)   # n x 8 (dihedral variants)

_POP8 = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def popcount64(x):
    """Number of set bits in each uint64 element."""
    x = x.astype(np.uint64)
    c = np.zeros(x.shape, dtype=np.uint8)
    for s in range(0, 64, 8):
        c += _POP8[((x >> np.uint64(s)) & np.uint64(255)).astype(np.int64)]
    return c

def groups(H, thr):
    n=len(H); parent=list(range(n))
    def f(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]; a=parent[a]
        return a
    base=H[:,0]
    for i in range(n):
        # distance between image i (identity) and all variants of j (min over 8) for j>i
        d=popcount64(H[i+1:,:]^base[i]).min(axis=1)
        for j in np.nonzero(d<=thr)[0]:
            a,b=f(i),f(i+1+j)
            if a!=b: parent[a]=b
    return [f(i) for i in range(n)]
