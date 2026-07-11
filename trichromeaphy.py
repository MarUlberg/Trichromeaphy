#!/usr/bin/env python3
"""
Trichrome merger
- Drag-and-drop RGB (and optional W) images onto this script.
- Detects channels from .r/.g/.b/.w subextension (fallback: first letter).
- Aligns all channels to W (if present) otherwise G using ECC.
- Merges RGB.
- Optionally injects W as luminance (LRGB).
- Auto-levels image.
- Adjustable saturation.

Requires:
    pip install opencv-python numpy
"""

import os
import sys
import cv2
import numpy as np

SATURATION = 1.15
AUTO_LEVEL_CLIP = 0.5   # percent clipped at each end


def get_channel(path):
    """
    Determine the image channel.

    Preferred:
        image.r.jpg
        image.g.jpg
        image.b.jpg
        image.w.jpg
        image.l.jpg

    Fallback:
        rimage.jpg
        gimage.jpg
        bimage.jpg
        wimage.jpg
        limage.jpg

    .w and .l are treated as the same luminance channel.
    """

    base = os.path.basename(path)
    parts = base.split(".")

    # Preferred: subextension
    if len(parts) >= 3:
        ch = parts[-2].lower()

        if ch in ("r", "g", "b"):
            return ch

        if ch in ("w", "l"):
            return "w"

    # Fallback: first letter
    ch = base[0].lower()

    if ch in ("r", "g", "b"):
        return ch

    if ch in ("w", "l"):
        return "w"

    raise ValueError(
        f"Can't determine channel from '{base}'. "
        "Expected .r/.g/.b/.w/.l or filename beginning with r/g/b/w/l."
    )


def align(ref, img):
    ref32 = ref.astype(np.float32)/255.0
    img32 = img.astype(np.float32)/255.0

    warp = np.eye(2,3,dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        200,
        1e-6
    )
    try:
        cv2.findTransformECC(
            ref32, img32, warp,
            cv2.MOTION_TRANSLATION,
            criteria
        )
        return cv2.warpAffine(
            img,
            warp,
            (ref.shape[1], ref.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT
        )
    except cv2.error:
        print("Alignment failed for one channel; using original.")
        return img


def autolevel(img):
    out = np.empty_like(img)
    clip = AUTO_LEVEL_CLIP

    for c in range(3):
        ch = img[:,:,c]
        lo = np.percentile(ch, clip)
        hi = np.percentile(ch, 100-clip)
        if hi <= lo:
            out[:,:,c]=ch
            continue
        ch = (ch.astype(np.float32)-lo)*255.0/(hi-lo)
        out[:,:,c]=np.clip(ch,0,255).astype(np.uint8)
    return out


def adjust_saturation(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= factor
    hsv[:,:,1] = np.clip(hsv[:,:,1],0,255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def replace_luminance(bgr, lum):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    ycc[:,:,0] = lum
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def output_name(red_path, lrgb):
    parts = os.path.basename(red_path).split(".")
    if len(parts)>=3 and parts[-2].lower()=="r":
        parts[-2] = "lrgb" if lrgb else "rgb"
        return os.path.join(os.path.dirname(red_path),".".join(parts))
    return os.path.join(
        os.path.dirname(red_path),
        ("lrgb" if lrgb else "rgb")+os.path.basename(red_path)[1:]
    )


def main():
    if len(sys.argv) < 4:
        print("Drag RGB (and optional W) images onto this script.")
        input()
        return

    imgs={}
    red_path=None

    for p in sys.argv[1:]:
        ch=get_channel(p)
        if ch in imgs:
            raise RuntimeError(f"Duplicate {ch}")
        img=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Couldn't read {p}")
        imgs[ch]=img
        if ch=="r":
            red_path=p

    for c in ("r","g","b"):
        if c not in imgs:
            raise RuntimeError(f"Missing {c}")

    shapes={i.shape for i in imgs.values()}
    if len(shapes)!=1:
        raise RuntimeError("Images must have same dimensions.")

    ref = imgs["w"] if "w" in imgs else imgs["g"]

    for k in list(imgs.keys()):
        if imgs[k] is not ref:
            imgs[k]=align(ref, imgs[k])

    merged=cv2.merge((imgs["b"],imgs["g"],imgs["r"]))

    used=False
    if "w" in imgs:
        merged=replace_luminance(merged, imgs["w"])
        used=True

    merged=autolevel(merged)
    merged=adjust_saturation(merged,SATURATION)

    out=output_name(red_path,used)
    cv2.imwrite(out, merged,[cv2.IMWRITE_JPEG_QUALITY,98])
    print("Saved:",out)

    preview=merged
    h,w=preview.shape[:2]
    s=min(1400/w,1000/h,1)
    if s<1:
        preview=cv2.resize(preview,None,fx=s,fy=s)
    cv2.imshow("Result",preview)
    cv2.waitKey(0)

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print(e)
        input("Press Enter...")
