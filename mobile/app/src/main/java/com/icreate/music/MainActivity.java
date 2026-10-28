package com.icreate.music;

import android.os.Bundle;
import android.view.ViewGroup;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import androidx.appcompat.app.AppCompatActivity;

import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.net.InetSocketAddress;
import java.net.Socket;

/**
 * iCreate Music for Android.
 *
 * Starts the bundled Python engine (the same code as the desktop app) and
 * serves its UI on 127.0.0.1. Nothing leaves the device.
 */
public class MainActivity extends AppCompatActivity {

    private static final int PORT = 8000;
    private WebView web;
    private volatile String error = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        web = new WebView(this);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setMediaPlaybackRequiresUserGesture(false);
        s.setAllowFileAccess(false);
        web.setWebViewClient(new WebViewClient());
        web.setBackgroundColor(0xFF0B0D12);
        setContentView(web, new ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        web.loadDataWithBaseURL(null, splash(), "text/html", "utf-8", null);

        final String outDir = getFilesDir().getAbsolutePath() + "/songs";
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    if (!Python.isStarted()) {
                        Python.start(new AndroidPlatform(MainActivity.this));
                    }
                    Python.getInstance()
                            .getModule("icreate_music.mobile_server")
                            .callAttr("serve_mobile", outDir, PORT);
                } catch (Throwable t) {
                    error = String.valueOf(t);
                }
            }
        }).start();

        new Thread(new Runnable() {
            @Override
            public void run() {
                boolean up = false;
                for (int i = 0; i < 240 && !up; i++) {           // up to ~60s
                    up = portOpen();
                    if (!up) {
                        try { Thread.sleep(250); } catch (InterruptedException ignored) { }
                    }
                }
                final boolean ready = up;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (ready) {
                            web.loadUrl("http://127.0.0.1:" + PORT + "/");
                        } else {
                            String msg = error == null ? "Engine did not start in time." : error;
                            web.loadDataWithBaseURL(null, failed(msg), "text/html", "utf-8", null);
                        }
                    }
                });
            }
        }).start();
    }

    private boolean portOpen() {
        Socket sock = new Socket();
        try {
            sock.connect(new InetSocketAddress("127.0.0.1", PORT), 200);
            return true;
        } catch (Exception e) {
            return false;
        } finally {
            try { sock.close(); } catch (Exception ignored) { }
        }
    }

    @Override
    public void onBackPressed() {
        if (web != null && web.canGoBack()) {
            web.goBack();
        } else {
            super.onBackPressed();
        }
    }

    private static String splash() {
        return "<html><body style='margin:0;height:100vh;display:flex;align-items:center;"
                + "justify-content:center;background:#0b0d12;color:#8b96ad;"
                + "font-family:sans-serif;text-align:center'>"
                + "<div><div style='font-size:15px;color:#e8ecf4;font-weight:600;"
                + "margin-bottom:8px'>iCreate Music</div>"
                + "starting the synth engine&hellip;</div></body></html>";
    }

    private static String failed(String msg) {
        return "<html><body style='margin:0;padding:24px;background:#0b0d12;color:#ff6b6b;"
                + "font-family:monospace;font-size:13px;line-height:1.5'>"
                + "<b>Engine failed to start</b><br><br>"
                + msg.replace("<", "&lt;").replace("&", "&amp;")
                + "</body></html>";
    }
}
