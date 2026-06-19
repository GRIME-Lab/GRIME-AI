import os
import threading
import http.server
import socketserver
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
from PyQt5.QtCore import QUrl, QTimer, pyqtSignal


class OpenStreetMapWidget(QWidget):
    mapReady = pyqtSignal(bool)  # True if loaded, False if failed/timed out

    def __init__(self, parent=None, timeout_ms=10000):
        super().__init__(parent)
        self.view = QWebEngineView(self)

        # Ensure JS and remote access are enabled
        s = self.view.settings()
        s.setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        s.setAttribute(QWebEngineSettings.LocalStorageEnabled, True)
        s.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        s.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)

        # Console bridge for debugging
        self.view.page().javaScriptConsoleMessage = self._console_logger

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)
        self.setLayout(layout)

        # Readiness state and operation queue
        self._ready = False
        self._pending_ops = []  # queue of (fn, args, kwargs)

        # Hook loadFinished -> readiness; add timeout
        self.view.loadFinished.connect(self._on_loaded)
        QTimer.singleShot(timeout_ms, self._on_timeout)

        # Start local HTTP server so OSM tiles receive a valid Referer/Origin
        self._server = None
        self._server_thread = None
        self._server_port = self._start_tile_server()

        # Cleanup guard so server shutdown / file deletion only ever runs once,
        # regardless of whether it's triggered via closeEvent or aboutToQuit
        self._cleaned_up = False

        # Primary cleanup path: app-level quit signal. This fires on normal
        # application exit regardless of whether this widget is a top-level
        # window or embedded as a child/tab, since embedded widgets do not
        # reliably receive closeEvent when their parent window closes.
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._cleanup)

        # Start loading the map
        self._load_map()

    # --------------------------------------------------------------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------------------------------------------------------------
    def _console_logger(self, level, msg, line, source_id):
        prefix = {0: "[JS]", 1: "[JS-WARN]", 2: "[JS-ERROR]"}.get(level, "[JS]")
        print(f"{prefix} {msg} (line {line}) source: {source_id}")

    # --------------------------------------------------------------------------------------------------------------
    # Local HTTP server — gives Qt WebEngine a valid HTTP origin so OSM tiles load
    # --------------------------------------------------------------------------------------------------------------
    def _start_tile_server(self):
        """
        Serve the app's resource directory over http://127.0.0.1:<port>/.
        Binds to port 0 so the OS picks a free port automatically.
        Returns the chosen port number.
        """
        base_path = os.path.abspath(os.path.dirname(__file__))
        self._serve_root = os.path.normpath(os.path.join(base_path, ".."))
        serve_root = self._serve_root

        class _Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=serve_root, **kwargs)

            def log_message(self, fmt, *args):
                pass  # suppress access log noise

        self._server = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
        port = self._server.server_address[1]
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._server_thread.start()
        print(f"[OSM] Local tile server started on http://127.0.0.1:{port}")
        return port

    # --------------------------------------------------------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------------------------------------------------------
    def _cleanup(self):
        """
        Shuts down the local tile server and removes the generated map.html.
        Safe to call multiple times (e.g. from both aboutToQuit and closeEvent);
        only does work on the first call.
        """
        if self._cleaned_up:
            return
        self._cleaned_up = True

        if self._server is not None:
            self._server.shutdown()
            self._server = None

        map_html_path = os.path.join(self._serve_root, "map.html")
        if os.path.exists(map_html_path):
            try:
                os.remove(map_html_path)
            except OSError:
                pass  # non-fatal

    def closeEvent(self, event):
        self._cleanup()
        super().closeEvent(event)

    # --------------------------------------------------------------------------------------------------------------
    # Resource helpers
    # --------------------------------------------------------------------------------------------------------------
    def load_shapefile(self, folder, filename):
        base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, "../resources", "shape_files", folder, filename)

    def discover_icon_files(self, images_dir):
        icon_files = {}
        for fname in os.listdir(images_dir):
            if not fname.lower().endswith(".png"):
                continue
            if not fname.startswith("marker-icon-"):
                continue
            if "2x" in fname:  # skip retina versions
                continue
            if "shadow" in fname:  # skip shadow
                continue
            color_part = fname[len("marker-icon-"):-len(".png")]
            color_key = color_part.replace("-", "_")
            icon_files[color_key] = fname
        return icon_files

    def _build_icon_js(self, images_dir, file_url):
        """
        Explicitly define only the icons we know exist in resources/leaflet/images.
        """
        shadow_icon = os.path.join(images_dir, "marker-shadow.png")
        shadow_url = file_url(shadow_icon) if os.path.exists(shadow_icon) else "null"

        icon_files = self.discover_icon_files(images_dir)
        print(icon_files)

        icon_js_defs = ""
        for color, fname in icon_files.items():
            icon_path = os.path.join(images_dir, fname)

            if not os.path.exists(icon_path):
                print(f"Icon file missing for {color}: {icon_path}")
                continue

            icon_url = file_url(icon_path)

            print(f"Defining {color} icon -> {icon_url}")

            icon_js_defs += f"""
                window.{color}Icon = new L.Icon({{
                    iconUrl: '{icon_url}',
                    shadowUrl: { 'null' if shadow_url == 'null' else f"'{shadow_url}'" },
                    iconSize: [25, 41],
                    iconAnchor: [12, 41],
                    popupAnchor: [1, -34],
                    shadowSize: [41, 41]
                }});
            """
        return icon_js_defs

    # --------------------------------------------------------------------------------------------------------------
    # Readiness and queuing
    # --------------------------------------------------------------------------------------------------------------
    def _on_loaded(self, ok):
        if ok and not self._ready:
            self._ready = True
            # Flush queued operations
            for fn, args, kwargs in self._pending_ops:
                fn(*args, **kwargs)
            self._pending_ops.clear()
            self.mapReady.emit(True)
        elif not ok:
            self.mapReady.emit(False)

    def _on_timeout(self):
        if not self._ready:
            print("Map load timed out")
            self.mapReady.emit(False)

    def _queue_or_run(self, fn, *args, **kwargs):
        if not self._ready:
            self._pending_ops.append((fn, args, kwargs))
        else:
            fn(*args, **kwargs)

    # --------------------------------------------------------------------------------------------------------------
    # Public API: safe anytime (queued before ready, immediate after)
    # --------------------------------------------------------------------------------------------------------------
    def set_center(self, lat, lng, zoom=12, add_marker=False, label="", color="red"):
        def _impl(lat, lng, zoom, add_marker, label, color):
            safe_label = label.replace("'", "\\'")
            color_key = color.replace("-", "_")  # normalize for JS variable names
            js = f"""
                (function() {{
                    if (!window.map) {{ console.error('Map not ready yet'); return; }}
                    window.map.setView([{lat}, {lng}], {zoom});
                    if ({str(add_marker).lower()}) {{
                        var iconVar = window['{color_key}Icon'];
                        if (iconVar) {{
                            L.marker([{lat}, {lng}], {{icon: iconVar}})
                                .addTo(window.map).bindPopup('{safe_label}');
                        }} else {{
                            console.warn('Icon for color {color} not defined; using default.');
                            L.marker([{lat}, {lng}])
                                .addTo(window.map).bindPopup('{safe_label}');
                        }}
                    }}
                }})();
            """
            self.view.page().runJavaScript(js)
        self._queue_or_run(_impl, lat, lng, zoom, add_marker, label, color)

    def add_geojson(self, geojson_str):
        def _impl(geojson_str):
            safe_geojson = geojson_str.replace("\\", "\\\\").replace("'", "\\'")
            js = f"""
                (function() {{
                    if (!window.map) {{ console.error('Map not ready yet'); return; }}
                    try {{
                        var data = JSON.parse('{safe_geojson}');
                        L.geoJSON(data).addTo(window.map);
                        console.log('GeoJSON layer added.');
                    }} catch (e) {{
                        console.error('Failed to parse GeoJSON:', e);
                    }}
                }})();
            """
            self.view.page().runJavaScript(js)
        self._queue_or_run(_impl, geojson_str)

    def add_pin(self, lat, lng, color="blue", label=None):
        def _impl(lat, lng, color, label):
            label = (label or f"{color.capitalize()} Pin: {lat}, {lng}").replace("'", "\\'")
            js = f"""
                (function() {{
                    if (!window.map) {{ console.error('Map not ready yet'); return; }}
                    var iconVar = window['{color}Icon'];
                    if (!iconVar) {{
                        console.warn('Icon for color {color} not defined; using default.');
                        L.marker([{lat}, {lng}]).addTo(window.map).bindPopup('{label}');
                        return;
                    }}
                    L.marker([{lat}, {lng}], {{icon: iconVar}}).addTo(window.map).bindPopup('{label}');
                }})();
            """
            self.view.page().runJavaScript(js)
        self._queue_or_run(_impl, lat, lng, color, label)

    # --------------------------------------------------------------------------------------------------------------
    # Internal: build and load HTML
    # --------------------------------------------------------------------------------------------------------------
    def _load_map(self):
        base_path = os.path.abspath(os.path.dirname(__file__))
        leaflet_dir = os.path.join(base_path, "../resources", "leaflet")
        leaflet_css = os.path.join(leaflet_dir, "leaflet.css")
        leaflet_js = os.path.join(leaflet_dir, "leaflet.js")
        images_dir = os.path.join(leaflet_dir, "images")

        # Build HTTP-relative paths from the serve root (one level above base_path)
        serve_root = self._serve_root

        def http_url(p):
            rel = os.path.relpath(os.path.normpath(p), serve_root).replace(os.sep, "/")
            return f"http://127.0.0.1:{self._server_port}/{rel}"

        icon_js_defs = self._build_icon_js(images_dir, http_url)

        html = f"""<!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8"/>
            <title>OpenStreetMap Viewer</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="{http_url(leaflet_css)}"/>
            <style>html, body, #map {{ height: 100%; margin: 0; padding: 0; }}</style>
        </head>
        <body>
            <div id="map"></div>
            <script src="{http_url(leaflet_js)}"></script>
            <script>
                (function initWhenReady() {{
                    function ready() {{
                        if (typeof L === 'undefined') {{
                            console.error('Leaflet not loaded yet; retrying...');
                            return setTimeout(ready, 100);
                        }}
                        try {{
                            // Default view so tiles paint immediately
                            window.map = L.map('map').setView([0, 0], 2);

                            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                                maxZoom: 19,
                                attribution: '&copy; OpenStreetMap contributors'
                            }}).addTo(window.map);

                            {icon_js_defs}

                            // Force Leaflet to recalculate dimensions after being created in a tab/layout
                            setTimeout(function () {{
                                window.map.invalidateSize();
                            }}, 0);

                            console.log('Leaflet map initialized; awaiting operations.');
                        }} catch (e) {{
                            console.error('Map init failed:', e.message || e);
                        }}
                    }}
                    if (document.readyState === 'loading') {{
                        document.addEventListener('DOMContentLoaded', ready);
                    }} else {{
                        ready();
                    }}
                }})();
            </script>
        </body>
        </html>"""

        # Write HTML to disk inside the served directory so it has an HTTP origin
        map_html_path = os.path.join(serve_root, "map.html")
        with open(map_html_path, "w", encoding="utf-8") as f:
            f.write(html)

        self.view.load(QUrl(f"http://127.0.0.1:{self._server_port}/map.html"))
