# -*- coding: utf-8 -*-
"""
hafiza_olustur.py  -  YakupTV referans hafızası (v3)

Her fotoğraf için birden fazla ön-işleme varyantı dener, en iyi algılamayı seçer:
  orijinal | kenar boşluğu (pad) | büyütme | kontrast (CLAHE) | pad+CLAHE | yatay çevirme
  + hiçbiri tutmazsa düşük eşikli "son çare" detektörü.

Çıktılar:
  yakup_hafiza.json   -> landmark + vektör (canlı taraf vektörü landmark'tan yeniden hesaplar)
  debug_hafiza/*.jpg  -> her fotonun algılanan iskeleti (hangi foto neden sorunlu, buradan bak)
"""
import json
import os
import time

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import poz_ortak as po

MODEL_YOLU = "pose_landmarker_heavy.task"
FOTO_KLASORU = "yakup_fotolar"
CIKTI_JSON = "yakup_hafiza.json"
DEBUG_KLASORU = "debug_hafiza"
DEBUG_KAYDET = True


def detector_olustur(esik, kisi_sayisi=3):
    opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_YOLU),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=kisi_sayisi,
        min_pose_detection_confidence=esik,
        min_pose_presence_confidence=esik,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def clahe_uygula(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def varyant_uret(rgb):
    """(isim, goruntu, olcek, pad_sol, pad_ust, flip) listesi."""
    H, W = rgb.shape[:2]
    uzun = max(H, W)
    varyantlar = [("orijinal", rgb, 1.0, 0, 0, False)]

    # Kenara yapışık / kesik vücutlarda detektör kenar boşluğu verilince çok daha iyi çalışır
    p = int(0.25 * uzun)
    pad = cv2.copyMakeBorder(rgb, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    varyantlar.append(("pad", pad, 1.0, p, p, False))

    if uzun < 720:
        s = 1280.0 / uzun
        buyuk = cv2.resize(rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_CUBIC)
        varyantlar.append(("buyutme", buyuk, s, 0, 0, False))

    cl = clahe_uygula(rgb)
    varyantlar.append(("clahe", cl, 1.0, 0, 0, False))
    varyantlar.append(("pad+clahe", cv2.copyMakeBorder(cl, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0)),
                       1.0, p, p, False))

    varyantlar.append(("flip", np.ascontiguousarray(rgb[:, ::-1]), 1.0, 0, 0, True))
    return varyantlar


def geri_esle(lm, olcek, pl, pt, flip, W):
    """Varyant koordinatını orijinal foto piksel koordinatına çevir."""
    out = lm.copy()
    out[:, 0] = (out[:, 0] - pl) / olcek
    out[:, 1] = (out[:, 1] - pt) / olcek
    if flip:
        out = po.ayna(out, W)   # x geri çevrilir + sol/sağ etiketleri düzeltilir
    return out


def kalite(lm, W, H):
    g = po.gecerli_maske(lm, W, H)
    k = po.KALITE_NOKTALARI
    return float((lm[k, 2] * g[k]).sum() / len(k))


def en_iyi_algilama(detectors, rgb):
    H, W = rgb.shape[:2]
    en_iyi = None
    for det_ad, det in detectors:
        for ad, img, s, pl, pt, flip in varyant_uret(rgb):
            try:
                lm = po.poz_tespit(det, img)
            except Exception as e:
                print(f"    {ad}: hata {e}")
                continue
            if lm is None:
                continue
            lm = geri_esle(lm, s, pl, pt, flip, W)
            vek, bilgi = po.vektor_hesapla(lm, W, H)
            q = kalite(lm, W, H)
            puan = q + 0.01 * bilgi["gecerli"]   # eşitlikte daha çok öznitelik veren kazanır
            if bilgi["gecerli"] > 0 and (en_iyi is None or puan > en_iyi["puan"]):
                en_iyi = dict(lm=lm, vek=vek, bilgi=bilgi, kalite=q, puan=puan,
                              varyant=f"{det_ad}/{ad}")
        if en_iyi is not None and en_iyi["kalite"] >= 0.5:
            break   # normal detektör yeterli bulduysa son çareye gerek yok
    return en_iyi


def en_iyi_ek(yuz_det, jest_det, rgb):
    """Yüz/el için de varyant dene. Pad/büyütme oranları korur; el-yüz mesafesi
    yüz genişliğine normalize olduğu için koordinat geri çevirmeye gerek yok."""
    en_iyi = None
    for ad, img, s, pl, pt, flip in varyant_uret(rgb):
        if flip:
            continue
        try:
            v, b, js = po.ek_tespit(yuz_det, jest_det, img)
        except Exception as e:
            print(f"    ek/{ad}: hata {e}")
            continue
        puan = (1.0 if b["yuz"] else 0.0) + b["el"] + 0.5 * b["jest_skor"]
        if en_iyi is None or puan > en_iyi[0]:
            en_iyi = (puan, v, b, ad)
    return en_iyi


def debug_kaydet(dosya_adi, rgb, sonuc, ek_bilgi=None):
    os.makedirs(DEBUG_KLASORU, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    if sonuc is not None:
        po.iskelet_ciz(bgr, sonuc["lm"], W, H, kalinlik=max(2, W // 300))
        yazi = (f"{sonuc['varyant']} q={sonuc['kalite']:.2f} "
                f"oz={sonuc['bilgi']['gecerli']}/{len(po.OZELLIK_ISIMLERI)} "
                f"olcek={sonuc['bilgi']['olcek_kaynak']}")
        renk = (0, 255, 0)
    else:
        yazi, renk = "ALGILANAMADI", (0, 0, 255)
    if ek_bilgi is not None:
        yazi += f" | yuz={'var' if ek_bilgi['yuz'] else 'YOK'} el={ek_bilgi['el']} jest={ek_bilgi['jest']}"
        if sonuc is None and (ek_bilgi["yuz"] or ek_bilgi["el"]):
            renk = (0, 200, 255)
    fs = max(0.6, W / 1200)
    cv2.putText(bgr, yazi, (10, int(40 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), int(5 * fs))
    cv2.putText(bgr, yazi, (10, int(40 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, renk, max(1, int(2 * fs)))
    ok, buf = cv2.imencode(".jpg", bgr)
    if ok:
        buf.tofile(os.path.join(DEBUG_KLASORU, os.path.splitext(dosya_adi)[0] + ".jpg"))


def main():
    yollar = po.resim_listele(FOTO_KLASORU)
    if not yollar:
        print(f"'{FOTO_KLASORU}' içinde fotoğraf yok.")
        return
    print(f"{len(yollar)} fotoğraf bulundu.\n")

    detectors = [("normal", detector_olustur(0.3)), ("sonCare", detector_olustur(0.1))]
    yuz_det = po.yuz_detector(video=False, esik=0.3)
    jest_det = po.jest_detector(video=False, esik=0.3)

    kayitlar, basarisiz, zayif = {}, [], []
    t0 = time.time()
    for yol in yollar:
        dosya_adi = os.path.basename(yol)
        rgb = po.resim_yukle_rgb(yol)
        if rgb is None:
            basarisiz.append(dosya_adi)
            continue
        H, W = rgb.shape[:2]
        sonuc = en_iyi_algilama(detectors, rgb)
        ek = en_iyi_ek(yuz_det, jest_det, rgb)
        ek_v, ek_b = (ek[1], ek[2]) if ek else (np.full(len(po.EK_ISIMLERI), np.nan), None)
        if DEBUG_KAYDET:
            debug_kaydet(dosya_adi, rgb, sonuc, ek_b)

        yuz_el_var = ek_b is not None and (ek_b["yuz"] or ek_b["el"] > 0)
        if sonuc is None and not yuz_el_var:
            print(f"  [X] {dosya_adi}: ne poz ne yüz/el algılandı")
            basarisiz.append(dosya_adi)
            continue

        b = sonuc["bilgi"] if sonuc else {"gecerli": 0, "olcek_kaynak": None}
        kayitlar[dosya_adi] = {
            "genislik": W, "yukseklik": H,
            "varyant": sonuc["varyant"] if sonuc else None,
            "kalite": round(sonuc["kalite"], 4) if sonuc else 0.0,
            "olcek_kaynak": b["olcek_kaynak"],
            "gecerli_ozellik": b["gecerli"],
            "landmarks": ([[round(float(x), 2), round(float(y), 2), round(float(v), 4)]
                           for x, y, v in sonuc["lm"]] if sonuc else None),
            "vektor": ([None if not np.isfinite(x) else round(float(x), 5) for x in sonuc["vek"]]
                       if sonuc else None),
            "ek_vektor": [None if not np.isfinite(x) else round(float(x), 5) for x in ek_v],
            "ek_bilgi": ek_b,
        }
        isaret = "OK" if b["gecerli"] >= 12 or yuz_el_var else "ZAYIF"
        if isaret == "ZAYIF":
            zayif.append(dosya_adi)
        ekstr = (f"yüz={'var' if ek_b and ek_b['yuz'] else 'yok'} el={ek_b['el'] if ek_b else 0} "
                 f"jest={ek_b['jest'] if ek_b else None}")
        print(f"  [{isaret}] {dosya_adi}: poz={sonuc['varyant'] if sonuc else 'YOK'}  "
              f"öznitelik={b['gecerli']}/{len(po.OZELLIK_ISIMLERI)}  {ekstr}")

    veri = {
        "surum": po.FEATURE_VERSION,
        "olusturma": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ozellik_isimleri": po.OZELLIK_ISIMLERI,
        "ek_isimleri": po.EK_ISIMLERI,
        "kayitlar": kayitlar,
        "basarisiz": basarisiz,
    }
    with open(CIKTI_JSON, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, indent=1)

    print(f"\n{len(kayitlar)}/{len(yollar)} foto hafızaya eklendi ({time.time() - t0:.1f}s) -> {CIKTI_JSON}")
    if zayif:
        print(f"Az bilgi veren (kısmi vücut) fotolar: {', '.join(zayif)}")
    if basarisiz:
        print(f"ALGILANAMAYAN: {', '.join(basarisiz)}  -> '{DEBUG_KLASORU}/' içinden kontrol et")


if __name__ == "__main__":
    main()
