# iCreate Music — Android app

The same Python engine that runs on desktop, packaged as an APK. Chaquopy
bundles CPython 3.11 + numpy inside the app, so the phone generates music
on-device with no server and no network.

## What it does

1. `MainActivity` starts Python (Chaquopy) on a background thread.
2. `icreate_music.mobile_server.serve_mobile()` serves the normal web UI on
   `127.0.0.1:8000`.
3. A `WebView` loads it. Songs are written to the app's private files dir
   (`/data/data/com.icreate.music/files/songs`).

The UI is the identical single-page app from the desktop version — it is just
rendered in a WebView.

## Build it

**Option A — GitHub Actions (no setup).** Push to `main` (or run the *Build APK*
workflow manually from the Actions tab). The APK lands in the run's artifacts as
`icreate-music.apk`. This is the recommended route: the runner already has the
JDK, Android SDK and Gradle.

**Option B — Android Studio.** Open the `mobile/` folder, let Gradle sync, then
*Build → Build Bundle(s)/APK(s) → Build APK(s)*.

**Option C — command line**, with JDK 17 + Android SDK + Gradle 8.9 installed:

```bash
cp -r ../icreate_music app/src/main/python/icreate_music   # bundle the engine
gradle assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Notes and limits

* **First launch is slow** — Python and numpy unpack on first run; the splash
  screen can sit for 10–30 s. Later launches are much faster.
* **Rendering is slower than desktop.** Expect roughly 2–4× the desktop render
  time for the same length. Keep songs at 30–60 s on a phone.
* **Debug-signed.** The APK from CI is debug-signed, fine for sideloading, not
  for the Play Store. For release you need your own keystore
  (`assembleRelease` + `signingConfig`).
* **arm64-v8a and armeabi-v7a only** — that covers essentially every phone from
  the last decade. Emulators on x86 need `x86_64` added to `abiFilters`.
* Songs live inside the app's sandbox. Use the *download wav* link in the UI to
  save or share a track.
