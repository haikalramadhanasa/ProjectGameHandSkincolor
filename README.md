# project melempar fumo ke keranjang 

> Game fisika lempar objek interaktif berbasis Python dan **OpenCV**, di mana pemain mengontrol kekuatan (*charge*), sudut arah (*aim*), dan lemparan Fumo menggunakan gestur tangan dan objek berwarna secara *real-time* lewat webcam.

---

## feature

- **Hand & Color Tracking (OpenCV):** 
  - **Kotak Kontrol 1 (Kiri):** Gunakan punggung tangan untuk mengisi daya (*charge power* naik-turun secara dinamis) dan tunjukkan objek/kertas kuning untuk melakukan *release* (lempar).
  - **Kotak Kontrol 2 (Kanan):** Geser tangan secara horizontal untuk mengatur sudut lintasan lemparan (*trajectory offset*).
- **Simulasi Fisika Parabola & Pantulan:** Fumo bergerak mengikuti gravitasi, memantul saat membentur dinding batas jendela (*border bounce*), serta berputar secara dinamis (*dynamic rotation*) saat melayang.
- **Dynamic Moving Basket:** Keranjang bergerak ke kiri dan kanan dengan kecepatan yang semakin meningkat setiap kali kamu berhasil mencetak skor!
- **Victory & Game Over System:** Target skor hingga 3 poin untuk menyelesaikan permainan.

---

##  Kebutuhan Sistem & Dependensi

Pastikan pustaka Python berikut terinstall sebelum menjalankan game:

```bash
pip install opencv-python numpy


[![Demo video](https://img.youtube.com/vi/sUYIczWBZQQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=sUYIczWBZQQ)
