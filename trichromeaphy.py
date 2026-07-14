import os, sys
import cv2
import numpy as np

SATURATION = 1.15
AUTO_LEVEL_CLIP = 0.5

# Add decorative frame around 128x112 BMP images
ADD_FRAME = True
FRAME_COLOR = "frame-color.png"
FRAME_ORIGINAL = "frame-original.png"
FRAME_SEPIA = "frame-sepia.png"

# Adjust W coefficients to match your monochrome sensor if needed.
FILTERS = {
    "r": np.array([1,0,0], np.float32),
    "g": np.array([0,1,0], np.float32),
    "b": np.array([0,0,1], np.float32),

    "c": np.array([0,1,1], np.float32),
    "m": np.array([1,0,1], np.float32),
    "y": np.array([1,1,0], np.float32),

    # Calibrated from your images
    "w": np.array([0.2952,0.7040,0.0188], np.float32),
}

def get_channel(path):
    base = os.path.basename(path)
    parts = os.path.splitext(base)[0].split(".")

    # filename.l.bmp -> w
    # filename.r.bmp -> r
    # filename.bmp   -> w

    if len(parts) >= 2:
        ch = parts[-1].lower()
    else:
        ch = "w"

    if ch == "l":
        ch = "w"

    if ch not in FILTERS:
        # Unknown suffix? Treat it as luminance.
        ch = "w"

    return ch
    
def align(ref,img):
    if ref is img:
        return img
    warp=np.eye(2,3,dtype=np.float32)
    crit=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,150,1e-6)
    try:
        cv2.findTransformECC(ref.astype(np.float32)/255,
                             img.astype(np.float32)/255,
                             warp,
                             cv2.MOTION_TRANSLATION,
                             crit)
        return cv2.warpAffine(img,warp,(ref.shape[1],ref.shape[0]),
                              flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REFLECT)
    except cv2.error:
        return img

def apply_sepia_2bit(img):
    palette = np.array([
        [255, 255, 255],
        [ 55, 190, 225],
        [ 50, 115, 135],
        [  0,   0,   0],
    ], dtype=np.uint8)

    gray = img.astype(np.uint8)

    idx = np.zeros_like(gray, dtype=np.uint8)
    idx[gray <= 60] = 3          # black
    idx[(gray > 60) & (gray <= 156)] = 2   # brown (120)
    idx[(gray > 156) & (gray <= 223)] = 1  # gold (192)
    idx[gray > 223] = 0          # white

    return palette[idx]
    
def reconstruct(imgs):
    """
    Reconstruct an RGB image from any combination of filters.

    Supports:
        L
        R
        G
        B
        C
        M
        Y

    in any quantity (1-7 channels).

    If fewer than three independent color measurements are available,
    a least-squares reconstruction is performed and missing color
    information is approximated from luminance (or the first supplied
    image if luminance isn't present).
    """

    # ----------------------------------------------------------
    # Single-channel images
    # ----------------------------------------------------------

    if len(imgs) == 1:

        ch, img = next(iter(imgs.items()))

        # Luminance or any single filter:
        # simply display it using the sepia palette.
        return apply_sepia_2bit(img)
        
    chans = list(imgs.keys())

    A = np.stack([FILTERS[c] for c in chans]).astype(np.float32)

    h, w = next(iter(imgs.values())).shape

    stack = np.stack(
        [imgs[c] for c in chans],
        axis=-1
    ).astype(np.float32)

    # Solve using the Moore-Penrose pseudoinverse.
    # Works for any number of channels, even rank-deficient cases.
    pinv = np.linalg.pinv(A)

    rgb = stack.reshape(-1, len(chans)) @ pinv.T
    rgb = rgb.reshape(h, w, 3)

    # ----------------------------------------------------------
    # If the supplied filters cannot uniquely determine RGB,
    # approximate missing color information.
    # ----------------------------------------------------------

    if np.linalg.matrix_rank(A) < 3:

        if "w" in imgs:
            lum = imgs["w"].astype(np.float32)
        else:
            lum = next(iter(imgs.values())).astype(np.float32)

        # Which RGB components are unconstrained?
        constrained = np.any(np.abs(A) > 1e-6, axis=0)

        for i in range(3):
            if not constrained[i]:
                rgb[:, :, i] = lum

    rgb = np.clip(rgb, 0, 255)

    # ----------------------------------------------------------
    # Empirical correction for L+R+G (missing blue)
    # ----------------------------------------------------------

    if set(chans) == {"w", "r", "g"}:

        b = rgb[:, :, 2]

        b = (
            b * 1.35 +
            rgb[:, :, 1] * 0.12
        )

        b = cv2.GaussianBlur(b, (0, 0), 0.8)

        rgb[:, :, 2] = np.clip(b, 0, 255)

    # ----------------------------------------------------------
    # Empirical correction for L+G+B (missing red)
    # ----------------------------------------------------------

    elif set(chans) == {"w", "g", "b"}:

        r = (
            rgb[:, :, 0] * 1.12 +
            rgb[:, :, 1] * 0.05
        )

        rgb[:, :, 0] = np.clip(r, 0, 255)

    # ----------------------------------------------------------
    # Empirical correction for L+R+B (missing green)
    # ----------------------------------------------------------

    elif set(chans) == {"w", "r", "b"}:

        rgb[:, :, 1] *= 0.97
        rgb[:, :, 1] = np.clip(rgb[:, :, 1], 0, 255)

    return cv2.merge((
        rgb[:, :, 2].astype(np.uint8),
        rgb[:, :, 1].astype(np.uint8),
        rgb[:, :, 0].astype(np.uint8),
    ))
    
def replace_lum(bgr,lum):
    ycc=cv2.cvtColor(bgr,cv2.COLOR_BGR2YCrCb)
    ycc[:,:,0]=lum
    return cv2.cvtColor(ycc,cv2.COLOR_YCrCb2BGR)

def autolevel(img):
    out=img.copy()
    for c in range(3):
        lo=np.percentile(out[:,:,c],AUTO_LEVEL_CLIP)
        hi=np.percentile(out[:,:,c],100-AUTO_LEVEL_CLIP)
        if hi>lo:
            x=(out[:,:,c].astype(np.float32)-lo)*255/(hi-lo)
            out[:,:,c]=np.clip(x,0,255)
    return out.astype(np.uint8)

def sat(img,f):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1]=np.clip(hsv[:,:,1]*f,0,255)
    return cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)

def main():
    if len(sys.argv) < 2:
        print("Drop image files onto this script.")
        input()
        return

    # ----------------------------------------------------------
    # If every dropped image has no channel suffix,
    # process each one independently as a Game Boy image.
    # ----------------------------------------------------------

    def has_channel(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split(".")
        return (
            len(parts) >= 2
            and parts[-1].lower() in ("l", "r", "g", "b", "c", "m", "y")
        )

    if len(sys.argv) > 2 and not any(has_channel(p) for p in sys.argv[1:]):

        for first in sys.argv[1:]:

            img = cv2.imread(first, 0)
            if img is None:
                continue

            bgr = apply_sepia_2bit(img)
            bgr = autolevel(bgr)
            bgr = sat(bgr, SATURATION)

            base = os.path.basename(first)
            stem, ext = os.path.splitext(base)

            out = os.path.join(
                os.path.dirname(first),
                stem + ".gb" + ext
            )

            if (
                ADD_FRAME
                and first.lower().endswith(".bmp")
                and bgr.shape[1] == 128
                and bgr.shape[0] == 112
            ):
                frame = cv2.imread(
                    os.path.join(os.path.dirname(__file__), FRAME_SEPIA),
                    cv2.IMREAD_COLOR
                )

                if frame is not None:
                    framed = frame.copy()
                    framed[16:128, 16:144] = bgr
                    bgr = framed

            cv2.imwrite(out, bgr, [cv2.IMWRITE_JPEG_QUALITY, 98])
            print("Saved", out)

        return
        
    imgs = {}
    first = None

    for p in sys.argv[1:]:
        ch = get_channel(p)
        im = cv2.imread(p, 0)
        if im is None:
            continue
        imgs[ch] = im
        if first is None:
            first = p

    ref = imgs.get("w", imgs.get("g", next(iter(imgs.values()))))

    for k in list(imgs):
        imgs[k] = align(ref, imgs[k])

    bgr = reconstruct(imgs)

    if "w" in imgs:
        bgr = replace_lum(bgr, imgs["w"])

    bgr = autolevel(bgr)
    bgr = sat(bgr, SATURATION)

    order = ["l", "r", "g", "b", "c", "m", "y"]
    supplied = []
    for ch in order:
        if ch == "l":
            if "w" in imgs:
                supplied.append("l")
        elif ch in imgs:
            supplied.append(ch)

    suffix = "".join(supplied)
    base = os.path.basename(first)
    stem, ext = os.path.splitext(base)

    if len(imgs) == 1:
        # Single-image mode:
        # garfield.l.jpg -> garfield.gb.jpg
        # flower.r.png   -> flower.gb.png

        parts = stem.split(".")

        if len(parts) >= 2:
            parts[-1] = "gb"
            stem = ".".join(parts)
        else:
            stem = stem + ".gb"

        out = os.path.join(
            os.path.dirname(first),
            stem + ext
        )

    else:
        order = ["l", "r", "g", "b", "c", "m", "y"]
        supplied = []

        for ch in order:
            if ch == "l":
                if "w" in imgs:
                    supplied.append("l")
            elif ch in imgs:
                supplied.append(ch)

        suffix = "".join(supplied)

        parts = base.split(".")

        if len(parts) >= 3:
            parts[-2] = suffix
            out = ".".join(parts)
        else:
            stem, ext = os.path.splitext(base)
            out = f"{stem}.{suffix}{ext}"

        out = os.path.join(
            os.path.dirname(first),
            out
        )

    # ----------------------------------------------------------
    # Optional decorative frame
    # ----------------------------------------------------------

    if (
        ADD_FRAME
        and first.lower().endswith(".bmp")
        and bgr.shape[1] == 128
        and bgr.shape[0] == 112
    ):

        # Choose frame based on number of input images
        if len(imgs) == 1:
            frame_name = FRAME_SEPIA
        elif len(imgs) == 2:
            frame_name = FRAME_ORIGINAL
        else:
            frame_name = FRAME_COLOR

        frame_path = os.path.join(
            os.path.dirname(__file__),
            frame_name
        )

        if os.path.exists(frame_path):

            frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)

            if (
                frame is not None
                and frame.shape[1] >= 160
                and frame.shape[0] >= 144
            ):

                framed = frame.copy()
                framed[16:128, 16:144] = bgr
                bgr = framed

    cv2.imwrite(out, bgr, [cv2.IMWRITE_JPEG_QUALITY, 98])
    print("Saved", out)
    cv2.imshow("Result", bgr)
    cv2.waitKey(0)
    
if __name__=="__main__":
    main()
