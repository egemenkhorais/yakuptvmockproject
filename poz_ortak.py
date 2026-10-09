# -*- coding: utf-8 -*-
"""
poz_ortak.py  -  YakupTV Poz Motoru v3 (ORTAK MODÜL)

hafiza_olustur.py ve canli_eslestir.py ikisi de bunu import eder.
Öznitelik mantığı TEK yerde -> veritabanı ile canlı taraf asla kaymaz.

v2'ye göre temel farklar:
  * Piksel koordinatı kullanılır (x*w, y*h). v2'de normalize x/y kullanıldığı için
    16:9 webcam ile dikey telefon fotoğrafında açılar FARKLI çıkıyordu.
  * Her landmark için visibility/presence + kadraj içi kontrolü yapılır.
    Görünmeyen uzuvdan hesaplanan öznitelik NaN olur (uydurma değer üretilmez).
  * Mesafe sadece iki tarafta da geçerli olan özniteliklerle hesaplanır,
    eksikler için asimetrik ceza uygulanır -> kolu/eli kadrajda olmayan
    fotoğraflar da eşleşebilir.
  * Ölçek: omuz yoksa gövde / kalça / kafa (kulak-göz) ile yedeklenir.
  * Ayna (sol-sağ swap) desteği.
"""
import math
import os

import numpy as np

FEATURE_VERSION = 31

# ------------------------------------------------------------------ AYARLAR
VIS_ESIK = 0.5            # visibility/presence bunun altındaysa nokta "yok" sayılır
SINIR_PAYI = 0.03         # kadraj dışına bu oranın ötesinde taşan nokta "yok" sayılır
EKSIK_CEZA_REF_VAR_CANLI_YOK = 0.09   # fotoda var, kamerada yok -> kullanıcı o uzvu göstermeli
EKSIK_CEZA_CANLI_VAR_REF_YOK = 0.02   # kamerada var, fotoda yok -> foto bilgi veremiyor, hafif ceza
MIN_ORTAK_AGIRLIK = 3.0   # bundan az ortak bilgi varsa karşılaştırma yapılmaz (sadece baş+omuz foto ~3.5)

RESIM_UZANTILARI = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif")

# MediaPipe Pose 33 nokta - sol/sağ ayna çiftleri
AYNA_CIFTLERI = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
                 (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26),
                 (27, 28), (29, 30), (31, 32)]

ISKELET = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
           (11, 23), (12, 24), (23, 24), (15, 19), (16, 20), (0, 7), (0, 8)]
CIZIM_NOKTALARI = [0, 7, 8, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24]
KALITE_NOKTALARI = [0, 11, 12, 13, 14, 15, 16, 23, 24]

# el için yedek noktalar: bilek -> işaret parmağı -> serçe -> başparmak
SOL = dict(omuz=11, dirsek=13, el=(15, 19, 17, 21), kalca=23, kulak=7, karsi_omuz=12)
SAG = dict(omuz=12, dirsek=14, el=(16, 20, 18, 22), kalca=24, kulak=8, karsi_omuz=11)

# ------------------------------------------------------------------ ÖZNİTELİK TANIMI
_SPEC = []
for _s in ("sol", "sag"):
    _SPEC += [
        (f"{_s}_dirsek_aci", 1.5),
        (f"{_s}_koltuk_aci", 1.2),        # kalça-omuz-dirsek (kalça gerekir)
        (f"{_s}_omuz_hatti_aci", 1.2),    # karşı omuz-omuz-dirsek (kalça GEREKMEZ)
        (f"{_s}_ust_kol_x", 0.8), (f"{_s}_ust_kol_y", 0.8),
        (f"{_s}_on_kol_x", 0.8), (f"{_s}_on_kol_y", 0.8),
        (f"{_s}_el_burun", 1.0),
        (f"{_s}_el_kulak", 0.8),
        (f"{_s}_el_yukseklik", 1.2),
        (f"{_s}_el_yatay", 0.8),
        (f"{_s}_dirsek_yukseklik", 0.8),
    ]
_SPEC += [
    ("eller_arasi", 1.0),
    ("omuz_hatti_x", 0.6), ("omuz_hatti_y", 0.6),   # gövdenin dönüklüğü / omuz eğimi
    ("govde_x", 0.6), ("govde_y", 0.6),             # kalça->omuz yönü (eğilme)
    ("bas_egim", 0.6),
    ("bas_yaw", 0.6),
    ("bas_pitch", 0.5),
    ("burun_omuz", 0.6),
]
OZELLIK_ISIMLERI = [s[0] for s in _SPEC]
AGIRLIKLAR = np.array([s[1] for s in _SPEC], dtype=np.float64)
_IDX = {n: i for i, n in enumerate(OZELLIK_ISIMLERI)}


# ------------------------------------------------------------------ YARDIMCILAR
def _norm(v):
    return float(np.hypot(v[0], v[1]))


def _aci_n(a, b, c):
    """b tepe noktası, 0..1 (derece/180). Hesaplanamazsa None."""
    if a is None or b is None or c is None:
        return None
    ba, bc = a - b, c - b
    n = _norm(ba) * _norm(bc)
    if n < 1e-6:
        return None
    cos = float(np.clip(np.dot(ba, bc) / n, -1.0, 1.0))
    return math.degrees(math.acos(cos)) / 180.0


def _birim(a, b):
    """a->b birim vektörü (x, y) ya da None."""
    if a is None or b is None:
        return None
    v = b - a
    n = _norm(v)
    if n < 1e-6:
        return None
    return v / n


def _orta(a, b):
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def gecerli_maske(lm, w, h):
    """lm: (33,3) [x_px, y_px, guven]. Görünür VE kadraj içinde olanlar True."""
    mx, my = SINIR_PAYI * w, SINIR_PAYI * h
    x, y, v = lm[:, 0], lm[:, 1], lm[:, 2]
    return (v >= VIS_ESIK) & (x >= -mx) & (x <= w + mx) & (y >= -my) & (y <= h + my)


def ayna(lm, w):
    """Görüntüyü yatay çevirmiş gibi: x -> w-x ve sol/sağ indeksleri takas."""
    out = lm.copy()
    out[:, 0] = w - out[:, 0]
    for a, b in AYNA_CIFTLERI:
        out[[a, b]] = out[[b, a]]
    return out


def olcek_bul(P):
    so, sa = P(11), P(12)
    ko, ka = P(23), P(24)
    ku1, ku2 = P(7), P(8)
    g1, g2 = P(2), P(5)

    omuz = _norm(so - sa) if (so is not None and sa is not None) else None
    govde = None
    if so is not None and sa is not None and ko is not None and ka is not None:
        govde = _norm(_orta(so, sa) - _orta(ko, ka)) * 0.75
    kalca = _norm(ko - ka) * 1.4 if (ko is not None and ka is not None) else None
    if ku1 is not None and ku2 is not None:
        bas = _norm(ku1 - ku2) * 2.5
    elif g1 is not None and g2 is not None:
        bas = _norm(g1 - g2) * 6.0
    else:
        bas = None

    if omuz is not None and omuz > 1e-6:
        # yan dönükken omuz genişliği küçülür -> gövde/kafa ile alt sınır koy
        adaylar = [omuz]
        if govde:
            adaylar.append(govde)
        if bas:
            adaylar.append(bas * 0.6)
        return max(adaylar), "omuz"
    for deger, ad in ((govde, "govde"), (kalca, "kalca"), (bas, "bas")):
        if deger is not None and deger > 1e-6:
            return deger, ad
    return None, None


# ------------------------------------------------------------------ VEKTÖR
def vektor_hesapla(lm, w, h):
    """
    lm: (33,3) numpy [x_px, y_px, guven]
    return: (vektor[NaN=yok], bilgi_dict)
    """
    gec = gecerli_maske(lm, w, h)
    xy = lm[:, :2].astype(np.float64)

    def P(i):
        return xy[i] if gec[i] else None

    def P_ilk(idxs):
        for i in idxs:
            if gec[i]:
                return xy[i]
        return None

    v = np.full(len(OZELLIK_ISIMLERI), np.nan, dtype=np.float64)
    olcek, kaynak = olcek_bul(P)
    if olcek is None:
        return v, {"olcek_kaynak": None, "gecerli": 0, "gorunen_nokta": int(gec.sum())}

    def koy(isim, deger, lo=None, hi=None):
        if deger is None:
            return
        deger = float(deger)
        if not np.isfinite(deger):
            return
        if lo is not None:
            deger = max(lo, min(hi, deger))
        v[_IDX[isim]] = deger

    def mes(a, b):
        if a is None or b is None:
            return None
        return _norm(a - b) / olcek / 2.0

    burun = P(0)
    eller = {}
    for s, t in (("sol", SOL), ("sag", SAG)):
        omuz, dirsek, el = P(t["omuz"]), P(t["dirsek"]), P_ilk(t["el"])
        kalca, kulak, karsi = P(t["kalca"]), P(t["kulak"]), P(t["karsi_omuz"])
        eller[s] = el

        koy(f"{s}_dirsek_aci", _aci_n(omuz, dirsek, el))
        koy(f"{s}_koltuk_aci", _aci_n(kalca, omuz, dirsek))
        koy(f"{s}_omuz_hatti_aci", _aci_n(karsi, omuz, dirsek))

        u = _birim(omuz, dirsek)
        if u is not None:
            koy(f"{s}_ust_kol_x", u[0]); koy(f"{s}_ust_kol_y", u[1])
        u = _birim(dirsek, el)
        if u is not None:
            koy(f"{s}_on_kol_x", u[0]); koy(f"{s}_on_kol_y", u[1])

        koy(f"{s}_el_burun", mes(el, burun), 0, 2.5)
        koy(f"{s}_el_kulak", mes(el, kulak), 0, 2.5)
        if el is not None and omuz is not None:
            koy(f"{s}_el_yukseklik", (el[1] - omuz[1]) / olcek / 2.0, -2, 2)
            koy(f"{s}_el_yatay", (el[0] - omuz[0]) / olcek / 2.0, -2, 2)
        if dirsek is not None and omuz is not None:
            koy(f"{s}_dirsek_yukseklik", (dirsek[1] - omuz[1]) / olcek / 2.0, -2, 2)

    koy("eller_arasi", mes(eller["sol"], eller["sag"]), 0, 2.5)

    so, sa = P(11), P(12)
    u = _birim(sa, so)
    if u is not None:
        koy("omuz_hatti_x", u[0]); koy("omuz_hatti_y", u[1])
    u = _birim(_orta(P(23), P(24)), _orta(so, sa))
    if u is not None:
        koy("govde_x", u[0]); koy("govde_y", u[1])

    # kafa
    k1, k2 = P(7), P(8)
    g1, g2 = P(2), P(5)
    if k1 is not None and k2 is not None:
        bas_sol, bas_sag = k1, k2
    elif g1 is not None and g2 is not None:
        bas_sol, bas_sag = g1, g2
    else:
        bas_sol = bas_sag = None
    if bas_sol is not None:
        u = _birim(bas_sag, bas_sol)
        if u is not None:
            koy("bas_egim", u[1])
        orta = _orta(bas_sol, bas_sag)
        gen = _norm(bas_sol - bas_sag)
        if burun is not None and gen > 1e-6:
            koy("bas_yaw", (burun[0] - orta[0]) / gen, -1.5, 1.5)
            koy("bas_pitch", (burun[1] - orta[1]) / olcek, -1, 1)
    if burun is not None and so is not None and sa is not None:
        koy("burun_omuz", (burun[1] - _orta(so, sa)[1]) / olcek / 2.0, -2, 2)

    return v, {"olcek_kaynak": kaynak,
               "gecerli": int(np.isfinite(v).sum()),
               "gorunen_nokta": int(gec.sum())}


# ------------------------------------------------------------------ MESAFE
def poz_mesafesi(canli, ref, w=None):
    """
    Ağırlıklı RMS fark. Küçük = benzer. return (skor, kapsama 0..1)
    canli ile ref yer DEĞİŞTİRMEMELİ (cezalar asimetrik).
    """
    if w is None:
        w = AGIRLIKLAR if len(canli) == len(AGIRLIKLAR) else TUM_AGIRLIK
    vc, vr = np.isfinite(canli), np.isfinite(ref)
    ortak = vc & vr
    birlesim = vc | vr
    w_ortak = w[ortak].sum()
    if w_ortak < MIN_ORTAK_AGIRLIK:
        return float("inf"), 0.0
    d2 = float((w[ortak] * (canli[ortak] - ref[ortak]) ** 2).sum())
    ceza = (EKSIK_CEZA_REF_VAR_CANLI_YOK * w[vr & ~vc].sum()
            + EKSIK_CEZA_CANLI_VAR_REF_YOK * w[vc & ~vr].sum())
    toplam = w[birlesim].sum()
    return math.sqrt((d2 + ceza) / toplam), float(w_ortak / toplam)


# ------------------------------------------------------------------ G/Ç
def resim_yukle_rgb(yol):
    """EXIF yönünü düzelterek RGB numpy döndürür. (Telefondan gelen yan dönük
    fotoğrafların algılanmama sebebi çoğunlukla budur.) Türkçe karakterli yol sorunsuz."""
    from PIL import Image, ImageOps
    if yol.lower().endswith((".heic", ".heif")):
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            print(f"  [!] HEIC için 'pip install pillow-heif' gerekli: {os.path.basename(yol)}")
            return None
    try:
        with Image.open(yol) as im:
            im = ImageOps.exif_transpose(im)
            return np.array(im.convert("RGB"))
    except Exception as e:
        print(f"  [!] Okunamadı: {os.path.basename(yol)} -> {e}")
        return None


def resim_listele(klasor):
    """glob('*.jpg') büyük harfli .JPG/.PNG'leri kaçırıyordu; bu kaçırmaz."""
    if not os.path.isdir(klasor):
        return []
    return sorted(os.path.join(klasor, f) for f in os.listdir(klasor)
                  if f.lower().endswith(RESIM_UZANTILARI) and not f.startswith("."))


def landmark_cevir(pose_landmarks, w, h):
    arr = np.zeros((33, 3), dtype=np.float64)
    for i, p in enumerate(pose_landmarks[:33]):
        vis = getattr(p, "visibility", None)
        pres = getattr(p, "presence", None)
        vis = 1.0 if vis is None else float(vis)
        pres = 1.0 if pres is None else float(pres)
        arr[i] = (p.x * w, p.y * h, min(vis, pres))
    return arr


def kisi_sec(adaylar, w, h):
    """Birden fazla kişi varsa en büyük + en net olanı seç."""
    if not adaylar:
        return None
    if len(adaylar) == 1:
        return adaylar[0]

    def puan(lm):
        g = gecerli_maske(lm, w, h)
        if g.sum() < 2:
            return 0.0
        pts = lm[g, :2]
        alan = (np.ptp(pts[:, 0]) + 1) * (np.ptp(pts[:, 1]) + 1)
        return alan * float(lm[KALITE_NOKTALARI, 2].mean())
    return max(adaylar, key=puan)


def poz_tespit(detector, rgb, zaman_ms=None):
    """IMAGE modunda zaman_ms=None, VIDEO modunda ms timestamp ver."""
    import mediapipe as mp
    h, w = rgb.shape[:2]
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    sonuc = detector.detect(img) if zaman_ms is None else detector.detect_for_video(img, int(zaman_ms))
    if not sonuc.pose_landmarks:
        return None
    return kisi_sec([landmark_cevir(k, w, h) for k in sonuc.pose_landmarks], w, h)


def iskelet_ciz(img_bgr, lm, w, h, x_ayna=False, renk=(255, 255, 0), kalinlik=2):
    """Sadece geçerli noktaları çizer. x_ayna=True ise ekranda flip'li gösterim için."""
    import cv2
    gec = gecerli_maske(lm, w, h)

    def nokta(i):
        x = w - lm[i, 0] if x_ayna else lm[i, 0]
        return int(round(x)), int(round(lm[i, 1]))
    for a, b in ISKELET:
        if gec[a] and gec[b]:
            cv2.line(img_bgr, nokta(a), nokta(b), renk, kalinlik)
    for i in CIZIM_NOKTALARI:
        cv2.circle(img_bgr, nokta(i), max(3, kalinlik + 2),
                   (0, 0, 255) if gec[i] else (90, 90, 90), -1)
    return img_bgr


# =====================================================================
# v3.1  YÜZ + EL (JEST) KATMANI
# Pose modeli yakın plan selfie'de neredeyse sadece omuz+kafa görür.
# "düz bakış" vs "başparmak yukarı" gibi pozları ayıran asıl bilgi
# yüz ifadesi, kafa açısı ve EL JESTİdir -> bunları ayrı modellerle ölçüyoruz.
# =====================================================================
YUZ_MODEL = "face_landmarker.task"
JEST_MODEL = "gesture_recognizer.task"
MODEL_URL = {
    YUZ_MODEL: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    JEST_MODEL: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
}

GESTURE_ISIMLERI = ["Thumb_Up", "Thumb_Down", "Open_Palm", "Closed_Fist",
                    "Pointing_Up", "Victory", "ILoveYou"]
BLEND = [
    ("gulumseme", ("mouthSmileLeft", "mouthSmileRight"), 1.8),
    ("goz_kisma", ("eyeSquintLeft", "eyeSquintRight"), 0.8),
    ("goz_kirpma", ("eyeBlinkLeft", "eyeBlinkRight"), 0.6),
    ("agiz_acik", ("jawOpen",), 0.8),
    ("kas_catma", ("browDownLeft", "browDownRight"), 0.6),
    ("kas_kaldirma", ("browInnerUp",), 0.6),
    ("dudak_buzme", ("mouthPucker",), 0.5),
]
EK_SPEC = ([("el_sayisi", 3.0)]
           + [(f"jest_{g}", 2.5) for g in GESTURE_ISIMLERI]
           + [("el_yuz_mesafe", 1.2), ("el_yuz_dy", 1.0)]
           + [(b[0], b[2]) for b in BLEND]
           + [("yuz_yaw", 1.0), ("yuz_pitch", 0.8), ("yuz_roll", 1.2)])
EK_ISIMLERI = [s[0] for s in EK_SPEC]
EK_AGIRLIK = np.array([s[1] for s in EK_SPEC], dtype=np.float64)
_EIDX = {n: i for i, n in enumerate(EK_ISIMLERI)}
TUM_AGIRLIK = np.concatenate([AGIRLIKLAR, EK_AGIRLIK])
POZ_BOYUT = len(OZELLIK_ISIMLERI)


def model_hazirla(yol):
    """Model yoksa indirir (bir kere)."""
    if os.path.exists(yol):
        return yol
    url = MODEL_URL.get(os.path.basename(yol))
    if url is None:
        raise FileNotFoundError(yol)
    print(f"{yol} indiriliyor...")
    import shutil
    import ssl
    import subprocess
    import urllib.request

    gecici = yol + ".part"
    hatalar = []

    # 1) certifi sertifikalarıyla (macOS python.org Python'unda sistem sertifikası yok)
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=ctx, timeout=60) as r, open(gecici, "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(gecici, yol)
        return yol
    except Exception as e:
        hatalar.append(f"certifi: {e}")

    # 2) sistem curl'ü (macOS Anahtar Zinciri'ndeki şirket sertifikalarını kullanır)
    if shutil.which("curl"):
        try:
            subprocess.run(["curl", "-fsSL", "--retry", "2", "-o", gecici, url],
                           check=True, timeout=300)
            os.replace(gecici, yol)
            return yol
        except Exception as e:
            hatalar.append(f"curl: {e}")

    if os.path.exists(gecici):
        os.remove(gecici)
    raise SystemExit(f"{yol} indirilemedi ({' | '.join(hatalar)}).\n"
                     f"Tarayıcıdan indirip proje klasörüne '{os.path.basename(yol)}' adıyla koy:\n{url}")


def yuz_detector(video=False, esik=0.4):
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision as mv
    return mv.FaceLandmarker.create_from_options(mv.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=model_hazirla(YUZ_MODEL)),
        running_mode=mv.RunningMode.VIDEO if video else mv.RunningMode.IMAGE,
        num_faces=2,
        min_face_detection_confidence=esik,
        min_face_presence_confidence=esik,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True))


def jest_detector(video=False, esik=0.4):
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision as mv
    return mv.GestureRecognizer.create_from_options(mv.GestureRecognizerOptions(
        base_options=mpp.BaseOptions(model_asset_path=model_hazirla(JEST_MODEL)),
        running_mode=mv.RunningMode.VIDEO if video else mv.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=esik,
        min_hand_presence_confidence=esik))


def _matris_acilar(M):
    R = np.asarray(M, dtype=np.float64)[:3, :3]
    pitch = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2]))
    roll = math.atan2(R[1, 0], R[0, 0])
    k = math.pi / 2
    return (float(np.clip(yaw / k, -1, 1)), float(np.clip(pitch / k, -1, 1)),
            float(np.clip(roll / k, -1, 1)))


def ek_hesapla(yuz_sonuc, jest_sonuc, w, h):
    """
    Yüz + el öznitelikleri. Jest detektörü çalıştıysa el_sayisi/jest_* HER ZAMAN
    tanımlı (el yoksa 0) -> "el yok" da bir bilgi; düz poz ile başparmak pozu
    bu sayede ayrılır.
    return (vektor, bilgi)
    """
    v = np.full(len(EK_ISIMLERI), np.nan, dtype=np.float64)
    bilgi = {"el": 0, "jest": None, "jest_skor": 0.0, "yuz": False}

    eller = []
    if jest_sonuc is not None:
        hl = list(jest_sonuc.hand_landmarks or [])
        v[_EIDX["el_sayisi"]] = min(len(hl), 2) / 2.0
        for g in GESTURE_ISIMLERI:
            v[_EIDX[f"jest_{g}"]] = 0.0
        for kat in (jest_sonuc.gestures or []):
            if not kat:
                continue
            c = kat[0]
            if c.category_name in GESTURE_ISIMLERI:
                i = _EIDX[f"jest_{c.category_name}"]
                v[i] = max(v[i], float(c.score))
                if c.score > bilgi["jest_skor"]:
                    bilgi["jest"], bilgi["jest_skor"] = c.category_name, float(c.score)
        eller = [np.array([[p.x * w, p.y * h] for p in el]) for el in hl]
        bilgi["el"] = len(hl)

    if yuz_sonuc is not None and yuz_sonuc.face_landmarks:
        yuzler = yuz_sonuc.face_landmarks

        def gen(f):
            return abs(f[454].x - f[234].x) * w
        i = max(range(len(yuzler)), key=lambda k: gen(yuzler[k]))
        f = yuzler[i]
        bilgi["yuz"] = True

        if yuz_sonuc.face_blendshapes and i < len(yuz_sonuc.face_blendshapes):
            bs = {c.category_name: float(c.score) for c in yuz_sonuc.face_blendshapes[i]}
            for ad, anahtarlar, _ in BLEND:
                vals = [bs[a] for a in anahtarlar if a in bs]
                if vals:
                    v[_EIDX[ad]] = float(np.mean(vals))
        mats = getattr(yuz_sonuc, "facial_transformation_matrixes", None)
        if mats and i < len(mats):
            yaw, pitch, roll = _matris_acilar(mats[i])
            v[_EIDX["yuz_yaw"]], v[_EIDX["yuz_pitch"]], v[_EIDX["yuz_roll"]] = yaw, pitch, roll

        burun = np.array([f[1].x * w, f[1].y * h])
        yg = math.hypot((f[454].x - f[234].x) * w, (f[454].y - f[234].y) * h)
        if eller and yg > 1e-6:
            merkezler = [el.mean(axis=0) for el in eller]
            d = [float(np.hypot(*(m - burun))) / yg for m in merkezler]
            j = int(np.argmin(d))
            v[_EIDX["el_yuz_mesafe"]] = min(d[j], 4.0) / 2.0
            v[_EIDX["el_yuz_dy"]] = float(np.clip((merkezler[j][1] - burun[1]) / yg, -3, 3)) / 2.0
    return v, bilgi


def ek_ayna(v):
    """Yatay çevrilmiş görüntünün ek vektörü: yaw ve roll işaret değiştirir."""
    o = v.copy()
    for n in ("yuz_yaw", "yuz_roll"):
        o[_EIDX[n]] = -o[_EIDX[n]]
    return o


def ek_tespit(yuz_det, jest_det, rgb, zaman_ms=None):
    import mediapipe as mp
    h, w = rgb.shape[:2]
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    if zaman_ms is None:
        ys, js = yuz_det.detect(img), jest_det.recognize(img)
    else:
        ys, js = yuz_det.detect_for_video(img, int(zaman_ms)), jest_det.recognize_for_video(img, int(zaman_ms))
    v, b = ek_hesapla(ys, js, w, h)
    return v, b, js


def bos_poz():
    return np.full(POZ_BOYUT, np.nan, dtype=np.float64)
