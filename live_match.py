# -*- coding: utf-8 -*-
"""
canli_eslestir.py  -  YakupTV Poz Eşleme v3 (canlı)

  * VIDEO modu (kareler arası takip -> daha hızlı ve daha az titreme)
  * Kısmi vücut: kadrajda olmayan uzuvlar karşılaştırmaya katılmaz
  * Ayna toleransı (M tuşu): sağ/sol kol karıştırsan da eşleşir
  * Zamansal yumuşatma (EMA) + histerezis + N kare onayı -> eşleşme zıplamaz
  * İlk 3 aday + kapsama oranı ekranda

Tuşlar:  Q/ESC çık   M ayna toleransı aç/kapa   D debug paneli
"""
import json
import os
import sys
import time

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import poz_ortak as po

MODEL_YOLU = "pose_landmarker_heavy.task"   # yavaşsa: pose_landmarker_full.task
HAFIZA_JSON = "yakup_hafiza.json"
FOTO_KLASORU = "yakup_fotolar"
KAMERA = 0

ESIK_GIRIS = 0.20        # eşleşme başlatma (skor bunun altında olmalı)
ESIK_CIKIS = 0.27        # aktif eşleşmeyi bırakma (histerezis)
ONAY_KARE = 4            # aynı aday kaç kare üst üste kazanmalı
DEGISIM_FARKI = 0.03     # aktif eşleşmeden bu kadar daha iyi olan aday geçebilir
EMA_ALFA = 0.45          # 1=yumuşatma yok
KAYIP_KARE = 10          # kişi kaybolursa kaç kare sonra eşleşme düşsün
MIN_KAPSAMA = 0.35       # ortak bilgi oranı bundan düşükse aday sayılmaz
AYNA_TOLERANS = True

KUCUK_RESIM = 240


def letterbox(img, boyut):
    h, w = img.shape[:2]
    s = boyut / max(h, w)
    k = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    tuval = np.zeros((boyut, boyut, 3), dtype=np.uint8)
    y0, x0 = (boyut - k.shape[0]) // 2, (boyut - k.shape[1]) // 2
    tuval[y0:y0 + k.shape[0], x0:x0 + k.shape[1]] = k
    return tuval


def hafiza_yukle():
    if not os.path.exists(HAFIZA_JSON):
        sys.exit(f"{HAFIZA_JSON} yok. Önce hafiza_olustur.py çalıştır.")
    with open(HAFIZA_JSON, "r", encoding="utf-8") as f:
        veri = json.load(f)
    if not isinstance(veri, dict) or "kayitlar" not in veri or veri.get("surum", 0) < 31:
        sys.exit("Eski format hafıza. hafiza_olustur.py'yi yeniden çalıştır.")

    refler, resimler = {}, {}
    for ad, k in veri["kayitlar"].items():
        if k.get("landmarks"):
            lm = np.array(k["landmarks"], dtype=np.float64)
            # poz vektörünü landmark'tan yeniden hesapla -> poz_ortak değişse bile tutarlı
            poz_v = po.vektor_hesapla(lm, k["genislik"], k["yukseklik"])[0]
        else:
            poz_v = po.bos_poz()
        ek_v = np.array([np.nan if x is None else x for x in k["ek_vektor"]], dtype=np.float64)
        vek = np.concatenate([poz_v, ek_v])
        if not np.isfinite(vek).any():
            continue
        refler[ad] = vek
        rgb = po.resim_yukle_rgb(os.path.join(FOTO_KLASORU, ad))
        if rgb is not None:
            resimler[ad] = letterbox(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), KUCUK_RESIM)
    if not refler:
        sys.exit("Hafızada kullanılabilir poz yok.")
    print(f"{len(refler)} referans poz yüklendi.")
    return refler, resimler


def detector_olustur():
    opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_YOLU),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def yazi(img, metin, org, renk=(255, 255, 255), olcek=0.6, kalinlik=2):
    cv2.putText(img, metin, org, cv2.FONT_HERSHEY_SIMPLEX, olcek, (0, 0, 0), kalinlik + 3)
    cv2.putText(img, metin, org, cv2.FONT_HERSHEY_SIMPLEX, olcek, renk, kalinlik)


class Eslestirici:
    def __init__(self, isimler):
        self.isimler = list(isimler)
        self.ema = {k: None for k in self.isimler}
        self.kapsama = {k: 0.0 for k in self.isimler}
        self.aktif = None
        self.aday, self.aday_sayac = None, 0
        self.kayip = 0

    def guncelle(self, skorlar):
        """skorlar: {isim: (skor, kapsama)} ya da None (kişi yok)."""
        if skorlar is None:
            self.kayip += 1
            if self.kayip >= KAYIP_KARE:
                self.aktif, self.aday, self.aday_sayac = None, None, 0
                self.ema = {k: None for k in self.isimler}
            return
        self.kayip = 0
        for k, (s, kap) in skorlar.items():
            if kap < MIN_KAPSAMA or not np.isfinite(s):
                s = 9.0
            self.ema[k] = s if self.ema[k] is None else EMA_ALFA * s + (1 - EMA_ALFA) * self.ema[k]
            self.kapsama[k] = kap

        en_iyi = min(self.isimler, key=lambda k: self.ema[k])
        en_skor = self.ema[en_iyi]

        if self.aktif is not None and self.ema[self.aktif] > ESIK_CIKIS:
            self.aktif = None

        hedef = None
        if en_skor < ESIK_GIRIS:
            if self.aktif is None:
                hedef = en_iyi
            elif en_iyi != self.aktif and en_skor < self.ema[self.aktif] - DEGISIM_FARKI:
                hedef = en_iyi

        if hedef is None:
            self.aday, self.aday_sayac = None, 0
        elif hedef == self.aday:
            self.aday_sayac += 1
            if self.aday_sayac >= ONAY_KARE:
                self.aktif, self.aday, self.aday_sayac = hedef, None, 0
        else:
            self.aday, self.aday_sayac = hedef, 1

    def ilk(self, n=3):
        gecerli = [k for k in self.isimler if self.ema[k] is not None]
        return sorted(gecerli, key=lambda k: self.ema[k])[:n]


def main():
    global AYNA_TOLERANS
    refler, resimler = hafiza_yukle()
    detector = detector_olustur()
    yuz_det = po.yuz_detector(video=True, esik=0.5)
    jest_det = po.jest_detector(video=True, esik=0.5)
    esl = Eslestirici(refler.keys())

    cap = cv2.VideoCapture(KAMERA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    debug = True
    son_ts = -1
    fps, t_onceki = 0.0, time.perf_counter()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        # Algılama ham (çevrilmemiş) karede; ekranda ayna görüntüsü gösterilir.
        ts = int(time.monotonic() * 1000)
        if ts <= son_ts:
            ts = son_ts + 1
        son_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm = po.poz_tespit(detector, rgb, ts)
        ek_ham, ek_b, jest_sonuc = po.ek_tespit(yuz_det, jest_det, rgb, ts)

        ekran = cv2.flip(frame, 1)
        skorlar = None
        bilgi = None
        kisi_var = lm is not None or ek_b["yuz"] or ek_b["el"] > 0
        if kisi_var:
            if lm is not None:
                # ekrandaki görüntüye göre vektör (v2'deki davranışla aynı)
                poz_ekran, bilgi = po.vektor_hesapla(po.ayna(lm, w), w, h)
                poz_ham = po.vektor_hesapla(lm, w, h)[0]
                po.iskelet_ciz(ekran, lm, w, h, x_ayna=True)
            else:
                poz_ekran = poz_ham = po.bos_poz()
            v_ekran = np.concatenate([poz_ekran, po.ek_ayna(ek_ham)])
            v_ham = np.concatenate([poz_ham, ek_ham]) if AYNA_TOLERANS else None

            skorlar = {}
            for ad, ref in refler.items():
                s, kap = po.poz_mesafesi(v_ekran, ref)
                if v_ham is not None:
                    s2, kap2 = po.poz_mesafesi(v_ham, ref)
                    if s2 < s:
                        s, kap = s2, kap2
                skorlar[ad] = (s, kap)

            # el noktaları (ekrana aynalı)
            for el in (jest_sonuc.hand_landmarks or []):
                for p in el:
                    cv2.circle(ekran, (int((1 - p.x) * w), int(p.y * h)), 3, (255, 0, 255), -1)
        esl.guncelle(skorlar)

        # ---- ARAYÜZ
        if esl.aktif is not None:
            if esl.aktif in resimler and w > KUCUK_RESIM + 40:
                ekran[20:20 + KUCUK_RESIM, w - KUCUK_RESIM - 20:w - 20] = resimler[esl.aktif]
                cv2.rectangle(ekran, (w - KUCUK_RESIM - 20, 20), (w - 20, 20 + KUCUK_RESIM), (0, 255, 0), 3)
            yazi(ekran, f"Poz: {esl.aktif}", (30, 45), (0, 255, 0), 0.8)
        elif skorlar is None:
            yazi(ekran, "Kisi algilanamadi", (30, 45), (0, 0, 255), 0.8)
        elif esl.aday is not None:
            yazi(ekran, f"Yakalaniyor... {esl.aday} ({esl.aday_sayac}/{ONAY_KARE})", (30, 45), (0, 255, 255), 0.8)
        else:
            yazi(ekran, "Poza yaklasin...", (30, 45), (0, 165, 255), 0.8)

        if debug:
            y = 80
            for ad in esl.ilk(3):
                s = esl.ema[ad]
                renk = (0, 255, 0) if s < ESIK_GIRIS else (0, 255, 255) if s < ESIK_CIKIS else (200, 200, 200)
                yazi(ekran, f"{ad[:28]:28s} skor={s:.3f} kap={esl.kapsama[ad]:.2f}", (30, y), renk, 0.55, 1)
                y += 26
            if bilgi is not None:
                yazi(ekran, f"oznitelik={bilgi['gecerli']}/{len(po.OZELLIK_ISIMLERI)} "
                            f"olcek={bilgi['olcek_kaynak']}", (30, y), (255, 255, 255), 0.5, 1)
                y += 24
            g = ek_ham[po._EIDX["gulumseme"]]
            r = -ek_ham[po._EIDX["yuz_roll"]]
            yazi(ekran, f"yuz={'var' if ek_b['yuz'] else 'yok'} el={ek_b['el']} "
                        f"jest={ek_b['jest']}({ek_b['jest_skor']:.2f}) "
                        f"gulumseme={0 if np.isnan(g) else g:.2f} roll={0 if np.isnan(r) else r:+.2f}",
                 (30, y), (255, 200, 255), 0.5, 1)
            simdi = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, simdi - t_onceki))
            t_onceki = simdi
            yazi(ekran, f"FPS {fps:.1f} | Ayna toleransi: {'ACIK' if AYNA_TOLERANS else 'KAPALI'} (M)",
                 (30, h - 20), (255, 255, 255), 0.5, 1)

        cv2.imshow("YakupTV Poz Esleme v3.1", ekran)
        tus = cv2.waitKey(1) & 0xFF
        if tus in (ord("q"), 27):
            break
        if tus == ord("m"):
            AYNA_TOLERANS = not AYNA_TOLERANS
        if tus == ord("d"):
            debug = not debug

    cap.release()
    detector.close()
    yuz_det.close()
    jest_det.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
