import os
import threading
import http.server
import socketserver
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings, QWebEnginePage
from PyQt5.QtCore import QUrl, QTimer, pyqtSignal
from PyQt5.QtGui import QDesktopServices
import json
from PyQt5.QtCore import QObject, pyqtSlot, QStandardPaths
from PyQt5.QtWebChannel import QWebChannel


class _ExternalLinkPage(QWebEnginePage):
    """Opens clicked hyperlinks in the system browser instead of the map view."""
    def acceptNavigationRequest(self, url, nav_type, is_main_frame):
        if nav_type == QWebEnginePage.NavigationTypeLinkClicked:
            QDesktopServices.openUrl(url)
            return False
        return super().acceptNavigationRequest(url, nav_type, is_main_frame)


class _MapBridge(QObject):
    """Receives map events from JavaScript over QWebChannel and re-emits them on the widget."""

    def __init__(self, widget):
        super().__init__(widget)
        self._widget = widget

    @pyqtSlot(str)
    def markerClicked(self, marker_id):
        self._widget.markerClicked.emit(marker_id)

    @pyqtSlot(float, float, float, float)
    def boundsChanged(self, south, west, north, east):
        self._widget.boundsChanged.emit(south, west, north, east)

    @pyqtSlot(str, bool)
    def overlayToggled(self, name, visible):
        self._widget._on_overlay_toggled(name, visible)

    @pyqtSlot(str)
    def boxSelected(self, ids_json):
        try:
            ids = json.loads(ids_json)
        except ValueError:
            ids = []
        self._widget.boxSelected.emit(ids)


class OpenStreetMapWidget(QWidget):
    mapReady = pyqtSignal(bool)  # True if loaded, False if failed/timed out
    markerClicked = pyqtSignal(str)                       # id of a pin added via add_marker_layer
    boundsChanged = pyqtSignal(float, float, float, float)  # south, west, north, east (after enable_bounds_events)
    boxSelected = pyqtSignal(list)                        # ids inside a shift+drag rectangle (after enable_box_select)
    overlayToggled = pyqtSignal(str, bool)                # a pin group was switched on/off in the map's layer panel

    # Pin groups listed in the map's layer panel, in this order. Pins added with group=<name> belong to it.
    OVERLAY_ORDER = ("MESONET", "PhenoCam", "USGS", "NEON")
    OVERLAY_SETTINGS_FILE = "map_layers.json"             # remembers which groups are shown, per user

    # Clustering (Leaflet.markercluster): each network clusters only with itself; clusters split apart on zoom-in.
    CLUSTER_RADIUS_PX = 60          # pins closer than this (screen pixels) merge into one cluster
    CLUSTER_OFF_AT_ZOOM = 11        # at this zoom level and closer, only overlapping pins stay merged...
    CLUSTER_OVERLAP_PX = 2          # ...pins within this many pixels; clicking such a cluster fans them out

    def __init__(self, parent=None, timeout_ms=10000, bounds_debounce_ms=0):
        self._overlay_visible = self._load_overlay_settings()
        super().__init__(parent)
        self.view = QWebEngineView(self)
        self.view.setPage(_ExternalLinkPage(self.view))

        # Ensure JS and remote access are enabled
        s = self.view.settings()
        s.setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        s.setAttribute(QWebEngineSettings.LocalStorageEnabled, True)
        s.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        s.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)

        # Console bridge for debugging
        self.view.page().javaScriptConsoleMessage = self._console_logger

        # JavaScript -> Python event channel (used only by the marker-layer / box-select / bounds API)
        self._bounds_debounce_ms = bounds_debounce_ms
        self._bridge = _MapBridge(self)
        self._channel = QWebChannel(self.view.page())
        self._channel.registerObject("mapBridge", self._bridge)
        self.view.page().setWebChannel(self._channel)

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

    def add_sdmesonet_pins(self, df, group=None):
        """Drop a blue pin per SD Mesonet station; popup shows full station info."""
        if df is None or df.empty:
            return
        from urllib.parse import urljoin
        base = "https://climate.sdstate.edu/information/stations/"
        for _, r in df.iterrows():
            lat, lon = r.get("lat"), r.get("lon")
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                continue

            station = r.get("station") or "SD Mesonet"
            url = r.get("station_url") or ""
            if url:
                station_html = f'<a href="{urljoin(base, url)}" title="Open SD Mesonet page">{station}</a>'
            else:
                station_html = str(station)

            fields = [
                ("Station", station_html),
                ("NWSLI", r.get("nwsli")),
                ("Detail", r.get("detail")),
                ("County", r.get("county")),
                ("Start", r.get("start")),
                ("Lat", r.get("lat")),
                ("Lon", r.get("lon")),
                ("Elv(Feet)", r.get("elv_ft")),
                ("Time Zone", r.get("utc_offset")),
                ("Camera", "Active" if r.get("active") else "Inactive"),
            ]
            label = "<br>".join(
                f"<b>{k}:</b> {v}" for k, v in fields
                if v not in ("", None) and str(v) != "nan"
            )
            self.add_pin(lat, lon, color="blue", label=label, group=group, cluster="SD MESONET")

    def add_nemesonet_pins(self, df, icon_svg=None, group=None):
        """Drop a pin per NE Mesonet station; popup shows full station info.
        icon_svg: draw each station with this SVG icon (the ear of corn); None keeps the standard red pin."""
        if df is None or df.empty:
            return
        import html
        markers = []
        for _, r in df.iterrows():
            lat, lon = r.get("lat"), r.get("lon")
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                continue

            station = html.escape(str(r.get("station") or "NE Mesonet"), quote=False)
            url = r.get("station_url") or ""
            if url:
                station_html = f'<a href="{url}" title="Open NE Mesonet page">{station}</a>'
            else:
                station_html = station

            elev_m, elev_ft = r.get("elevation_m"), r.get("elevation_ft")
            elevation = (f"{elev_m:.0f} m ({elev_ft:.0f} ft)"
                         if isinstance(elev_m, (int, float)) and isinstance(elev_ft, (int, float)) else "")

            fields = [
                ("Station", station_html),
                ("Station ID", r.get("station_id")),
                ("NWSLI", r.get("nwsli")),
                ("County", r.get("county")),
                ("NRD", r.get("nrd")),
                ("HUC6", r.get("huc6")),
                ("HUC8", r.get("huc8")),
                ("River Forecast Center", r.get("rfc_region")),
                ("Lat", r.get("lat")),
                ("Lon", r.get("lon")),
                ("Elevation", elevation),
                ("Time Zone", r.get("timezone")),
                ("10 m Temp/Humidity", "Yes" if r.get("has_10m_temp_humidity") else "No"),
                ("10 m Wind", "Yes" if r.get("has_10m_wind") else "No"),
                ("Soil Sensor Depths (in)", r.get("soil_sensor_depths_in")),
                ("Sponsored By", r.get("sponsored_by")),
            ]
            label = "<br>".join(
                f"<b>{k}:</b> {v if k == 'Station' else html.escape(str(v), quote=False)}"
                for k, v in fields
                if v not in ("", None) and str(v) != "nan"
            )
            if icon_svg:
                markers.append([float(lat), float(lon), label])
            else:
                self.add_pin(lat, lon, color="red", label=label, group=group, cluster="NE MESONET")
        if icon_svg and markers:
            self._add_svg_icon_markers("cornIcon", icon_svg, markers, group=group)

    def add_azmet_pins(self, df, icon_svg, group=None):
        """Drop a saguaro pin per AZMet station; popup shows station info (opens on click, like SD and NE)."""
        if df is None or df.empty:
            return
        import html
        markers = []
        for _, r in df.iterrows():
            lat, lon = r.get("lat"), r.get("lon")
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                continue
            station = html.escape(str(r.get("station") or "AZMet"), quote=False)
            url = r.get("station_url") or ""
            station_html = (f'<a href="{html.escape(url)}" title="Open AZMet station page">{station}</a>'
                            if url else station)
            fields = [
                ("Station", station_html),
                ("Symbol", r.get("symbol")),
                ("Station ID", r.get("station_id")),
                ("County", r.get("county")),
                ("Lat", lat),
                ("Lon", lon),
                ("Elevation", f"{r.get('elev_m')} m ({r.get('elev_ft')} ft)"),
                ("Status", r.get("status")),
                ("Data since", r.get("start_date")),
            ]
            label = "<br>".join(f"<b>{k}:</b> {v if k == 'Station' else html.escape(str(v), quote=False)}"
                                for k, v in fields if v not in ("", None) and str(v) != "nan")
            markers.append([float(lat), float(lon), label])
        self._add_svg_icon_markers("saguaroIcon", icon_svg, markers, group=group)

    def add_kansas_pins(self, df, icon_svg, group=None):
        """Drop a sunflower per Kansas Mesonet station; popup shows station info (opens on click)."""
        if df is None or df.empty:
            return
        import html
        markers = []
        for _, r in df.iterrows():
            lat, lon = r.get("lat"), r.get("lon")
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                continue
            fields = [
                ("Station", r.get("station")),
                ("Abbreviation", r.get("abbreviation")),
                ("County", r.get("county")),
                ("Network", r.get("network")),
                ("Lat", lat),
                ("Lon", lon),
                ("Elevation (m)", r.get("elevation_m")),
                ("5-minute data", f"{r.get('data_start_5min', '')} to {r.get('data_end_5min', '')}"
                                  if r.get("data_start_5min") not in ("", None) and str(r.get("data_start_5min")) != "nan"
                                  else ""),
            ]
            label = "<br>".join(f"<b>{k}:</b> {html.escape(str(v), quote=False)}"
                                for k, v in fields if v not in ("", None) and str(v) != "nan")
            url = r.get("station_url") or ""
            if url:   # the Kansas site has no per-station address; the station is chosen on that page
                label += (f'<br><a href="{html.escape(url)}" title="Open Kansas Mesonet station metadata">'
                          f'Station metadata</a> (choose {html.escape(str(r.get("station")), quote=False)} there)')
            markers.append([float(lat), float(lon), label])
        self._add_svg_icon_markers("sunflowerIcon", icon_svg, markers, size=(30, 30), anchor="center", group=group)

    def add_isusm_pins(self, df, icon_svg, group=None):
        """Drop a black-and-gold ear of corn per ISU Soil Moisture station; popup shows station info."""
        if df is None or df.empty:
            return
        import html
        markers = []
        for _, r in df.iterrows():
            lat, lon = r.get("lat"), r.get("lon")
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                continue
            station = html.escape(str(r.get("station") or "ISU Soil Moisture"), quote=False)
            url = r.get("station_url") or ""
            station_html = (f'<a href="{html.escape(url)}" title="Open IEM station page">{station}</a>'
                            if url else station)
            fields = [
                ("Station", station_html),
                ("Station ID", r.get("station_id")),
                ("County", r.get("county")),
                ("Lat", lat),
                ("Lon", lon),
                ("Elevation (m)", r.get("elevation_m")),
                ("Status", "Online" if r.get("online") else "Offline"),
                ("Data since", r.get("archive_begin")),
                ("Data until", r.get("archive_end")),
            ]
            label = "<br>".join(f"<b>{k}:</b> {v if k == 'Station' else html.escape(str(v), quote=False)}"
                                for k, v in fields if v not in ("", None) and str(v) != "nan")
            markers.append([float(lat), float(lon), label])
        self._add_svg_icon_markers("iowaCornIcon", icon_svg, markers, group=group)

    def _add_svg_icon_markers(self, icon_name, icon_svg, markers, size=(24, 40), anchor="bottom", group=None,
                              cluster=None):
        """
        Add markers drawn with an SVG icon; popups open on click.
        markers: [[lat, lon, popup_html], ...]. icon_name: JS name the icon is stored under (defined once).
        anchor: "bottom" puts the icon's bottom center on the station (standing icons such as the corn),
                "center" puts its middle on the station (round icons such as the sunflower).
        group:  name of a pin group in the map's layer panel; None adds the markers to the map directly.
        cluster: name of the cluster these markers join; defaults to icon_name, so each network clusters
                 only with itself. Clusters are used only when the markers belong to a group.
        """
        cluster = cluster or icon_name
        import json
        import urllib.parse
        icon_url = "data:image/svg+xml;charset=utf-8," + urllib.parse.quote(icon_svg)
        w, h = size
        anchor_y, popup_y = (h // 2, h // 2) if anchor == "center" else (h - 2, h - 6)

        def _impl(markers, icon_url):
            js = f"""
                (function() {{
                    if (!window.map) {{ console.error('Map not ready yet'); return; }}
                    if (!window[{json.dumps(icon_name)}]) {{
                        window[{json.dumps(icon_name)}] = L.icon({{iconUrl: {json.dumps(icon_url)},
                                                                   iconSize: [{w}, {h}],
                                                                   iconAnchor: [{w // 2}, {anchor_y}],
                                                                   popupAnchor: [0, -{popup_y}]}});
                    }}
                    var icon = window[{json.dumps(icon_name)}];
                    var target = {f"window.gaCluster({json.dumps(group)}, {json.dumps(cluster)}, icon)"
                                  if group else "window.map"};
                    var markers = {json.dumps(markers)};
                    markers.forEach(function(m) {{
                        if (!window.gaFirstTime({json.dumps((group or '') + '|' + cluster)}, m[0], m[1], m[2])) return;
                        L.marker([m[0], m[1]], {{icon: icon}}).addTo(target).bindPopup(m[2]);
                    }});
                }})();
            """
            self.view.page().runJavaScript(js)
        self._queue_or_run(_impl, markers, icon_url)

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

    def add_pin(self, lat, lng, color="blue", label=None, group=None, cluster=None):
        """
        group:   name of a pin group in the map's layer panel (e.g. "USGS"); None adds the pin to the map directly.
        cluster: name of the cluster this pin joins (one per network/source). Defaults to the group name, so each
                 group's pins cluster only with each other. Ungrouped pins without a cluster are never clustered.
        """
        def _impl(lat, lng, color, label, group, cluster):
            label = (label or f"{color.capitalize()} Pin: {lat}, {lng}").replace("'", "\\'")
            cluster = cluster or group
            js = f"""
                (function() {{
                    if (!window.map) {{ console.error('Map not ready yet'); return; }}
                    var iconVar = window['{color}Icon'];
                    if (!iconVar) {{
                        console.warn('Icon for color {color} not defined; using default.');
                    }}
                    var target = {f"window.gaCluster({json.dumps(group)}, {json.dumps(cluster)}, iconVar)"
                                  if cluster else "window.map"};
                    if (!window.gaFirstTime({json.dumps((group or '') + '|' + (cluster or 'map'))}, {lat}, {lng}, '{label}')) return;
                    var opts = iconVar ? {{icon: iconVar}} : {{}};
                    L.marker([{lat}, {lng}], opts).addTo(target).bindPopup('{label}');
                }})();
            """
            self.view.page().runJavaScript(js)
        self._queue_or_run(_impl, lat, lng, color, label, group, cluster)

    # --------------------------------------------------------------------------------------------------------------
    # Public API: interactive marker layers (clickable, hover tooltips, removable, recolorable)
    # --------------------------------------------------------------------------------------------------------------
    def add_marker_layer(self, name, markers, dot=None):
        """
        Add many pins in one call as a named, removable layer. Replaces any layer with the same name.

        markers: list of dicts with keys
            id       str    returned by markerClicked / boxSelected
            lat, lng float
            tooltip  str    HTML shown on hover (optional)
            color    str    pins: icon color key, e.g. "blue", "red" (default Leaflet icon if missing)
                            dots: CSS fill color
            border   str    dots only: CSS border color (optional; defaults to color)

        dot: None draws icon pins. A dict draws small circles instead:
            {"radius": px, "weight": border width px, "fill_opacity": 0..1}

        Clicking a pin or dot emits markerClicked(id).
        """
        def _impl(name, markers, dot):
            js = f"window.gaAddMarkerLayer({json.dumps(name)}, {json.dumps(markers)}, {json.dumps(dot)});"
            self.view.page().runJavaScript(js)
        self._queue_or_run(_impl, name, list(markers), dot)

    def clear_marker_layer(self, name):
        """Remove a layer added with add_marker_layer."""
        def _impl(name):
            self.view.page().runJavaScript(f"window.gaClearMarkerLayer({json.dumps(name)});")
        self._queue_or_run(_impl, name)

    def set_marker_color(self, layer, marker_id, color, border=None):
        """Change one pin's icon color (pins) or fill and border color (dots), e.g. to show it as selected."""
        def _impl(layer, marker_id, color, border):
            self.view.page().runJavaScript(
                f"window.gaSetMarkerColor({json.dumps(layer)}, {json.dumps(str(marker_id))}, "
                f"{json.dumps(color)}, {json.dumps(border)});")
        self._queue_or_run(_impl, layer, marker_id, color, border)

    def enable_box_select(self, enabled=True):
        """
        Shift+drag draws a selection rectangle and emits boxSelected(ids) for pins in
        marker layers inside it. While enabled, Leaflet's shift+drag zoom is turned off.
        """
        def _impl(enabled):
            self.view.page().runJavaScript(f"window.gaSetBoxSelect({json.dumps(bool(enabled))});")
        self._queue_or_run(_impl, enabled)

    def enable_bounds_events(self, enabled=True):
        """Emit boundsChanged after each pan/zoom. Emits the current bounds immediately when enabled."""
        def _impl(enabled):
            self.view.page().runJavaScript(
                f"window.gaBoundsEnabled = {json.dumps(bool(enabled))}; if (window.gaBoundsEnabled) window.gaEmitBounds();")
        self._queue_or_run(_impl, enabled)

    # --------------------------------------------------------------------------------------------------------------
    # Pin groups: layer panel visibility, remembered per user
    # --------------------------------------------------------------------------------------------------------------
    def _overlay_settings_path(self):
        folder = QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)
        return os.path.join(folder, self.OVERLAY_SETTINGS_FILE)

    def _load_overlay_settings(self):
        try:
            with open(self._overlay_settings_path(), "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return {str(k): bool(v) for k, v in data.items()}
        except Exception:
            return {}

    def _on_overlay_toggled(self, name, visible):
        self._overlay_visible[name] = bool(visible)
        try:
            path = self._overlay_settings_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self._overlay_visible, fh, indent=2)
        except OSError as e:
            print(f"Map layer choices not saved: {e}")
        self.overlayToggled.emit(name, bool(visible))

    def _event_js(self):
        """Page-side support for the marker-layer, box-select and bounds API."""
        js = """
            <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
            <script>
                (function () {
                    window.gaLayers = {};
                    window.gaBoundsEnabled = false;
                    window.gaBoxSelectEnabled = false;
                    var debounceMs = __DEBOUNCE_MS__;
                    var boundsTimer = null;

                    if (typeof QWebChannel !== 'undefined' && typeof qt !== 'undefined') {
                        new QWebChannel(qt.webChannelTransport, function (channel) {
                            window.gaBridge = channel.objects.mapBridge;
                            if (window.gaBoundsEnabled) window.gaEmitBounds();
                        });
                    } else {
                        console.warn('QWebChannel not available; map events will not reach Python.');
                    }

                    // window.map is the <div id="map"> element until Leaflet replaces it with the map object
                    function mapReady() {
                        return typeof L !== 'undefined' && window.map instanceof L.Map;
                    }

                    // Pin groups shown as checkboxes in Leaflet's layer panel. Visibility is remembered by Python.
                    window.gaOverlayOrder = __OVERLAY_ORDER__;
                    window.gaOverlayVisible = __OVERLAY_VISIBLE__;
                    window.gaOverlays = {};
                    window.gaLayerControl = null;
                    window.gaOverlay = function (name) {
                        if (window.gaOverlays[name]) return window.gaOverlays[name];
                        var group = L.layerGroup();
                        window.gaOverlays[name] = group;
                        if (!window.gaLayerControl) {
                            window.gaLayerControl = L.control.layers(null, {}, {
                                collapsed: false,
                                sortLayers: true,
                                sortFunction: function (a, b, nameA, nameB) {
                                    var ia = window.gaOverlayOrder.indexOf(nameA), ib = window.gaOverlayOrder.indexOf(nameB);
                                    return (ia < 0 ? 999 : ia) - (ib < 0 ? 999 : ib);
                                }
                            }).addTo(window.map);
                            window.map.on('overlayadd', function (e) {
                                if (window.gaBridge) window.gaBridge.overlayToggled(e.name, true);
                            });
                            window.map.on('overlayremove', function (e) {
                                if (window.gaBridge) window.gaBridge.overlayToggled(e.name, false);
                            });
                        }
                        window.gaLayerControl.addOverlay(group, name);
                        if (window.gaOverlayVisible[name] !== false) group.addTo(window.map);
                        return group;
                    };

                    // One marker cluster per network/source, placed inside its layer-panel group. The cluster icon
                    // is that network's own icon with the station count on it. Without the plugin, pins are
                    // simply added to the group.
                    window.gaClusters = {};
                    window.gaSeen = {};
                    // True the first time a pin (same cluster, position and popup) is added; repeats are skipped.
                    window.gaFirstTime = function (clusterId, lat, lng, popup) {
                        var k = clusterId + '|' + Number(lat).toFixed(6) + ',' + Number(lng).toFixed(6) + '|' + popup;
                        if (window.gaSeen[k]) return false;
                        window.gaSeen[k] = true;
                        return true;
                    };
                    window.gaClusterIcon = function (icon, count) {
                        var o = (icon && icon.options) || L.Icon.Default.prototype.options;
                        var url = o.iconUrl;
                        if (url && url.indexOf('data:') !== 0 && url.indexOf('/') < 0 && L.Icon.Default.imagePath) {
                            url = L.Icon.Default.imagePath + url;
                        }
                        var w = (o.iconSize || [25, 41])[0], h = (o.iconSize || [25, 41])[1];
                        var html = '<div class="ga-cluster" style="width:' + w + 'px;height:' + h + 'px">' +
                                   '<img src="' + url + '" style="width:' + w + 'px;height:' + h + 'px">' +
                                   '<span class="ga-count">' + count + '</span></div>';
                        return L.divIcon({html: html, className: 'ga-cluster-icon', iconSize: [w, h],
                                          iconAnchor: o.iconAnchor || [w / 2, h]});
                    };
                    window.gaCluster = function (groupName, key, icon) {
                        var parent = groupName ? window.gaOverlay(groupName) : window.map;
                        if (typeof L.markerClusterGroup !== 'function') return parent;
                        var id = (groupName || '') + '|' + key;
                        if (window.gaClusters[id]) return window.gaClusters[id];
                        var cg = L.markerClusterGroup({
                            maxClusterRadius: function (zoom) {
                                return zoom >= __CLUSTER_OFF_AT_ZOOM__ ? __CLUSTER_OVERLAP_PX__ : __CLUSTER_RADIUS_PX__;
                            },
                            spiderfyOnMaxZoom: true,
                            showCoverageOnHover: false,
                            iconCreateFunction: function (c) { return window.gaClusterIcon(icon, c.getChildCount()); }
                        });
                        if (groupName) parent.addLayer(cg); else cg.addTo(window.map);
                        window.gaClusters[id] = cg;
                        return cg;
                    };

                    window.gaIcon = function (color) {
                        return color ? window[String(color).replace(/-/g, '_') + 'Icon'] : null;
                    };

                    window.gaEmitBounds = function () {
                        if (!mapReady() || !window.gaBridge) return;
                        var b = window.map.getBounds();
                        window.gaBridge.boundsChanged(b.getSouth(), b.getWest(), b.getNorth(), b.getEast());
                    };

                    function onMoveEnd() {
                        if (!window.gaBoundsEnabled) return;
                        if (boundsTimer) clearTimeout(boundsTimer);
                        boundsTimer = setTimeout(window.gaEmitBounds, debounceMs);
                    }

                    window.gaClearMarkerLayer = function (name) {
                        var layer = window.gaLayers[name];
                        if (layer && mapReady()) window.map.removeLayer(layer.group);
                        delete window.gaLayers[name];
                    };

                    window.gaAddMarkerLayer = function (name, markers, dot) {
                        if (!mapReady()) { console.error('Map not ready yet'); return; }
                        window.gaClearMarkerLayer(name);
                        var group = L.layerGroup();
                        var byId = {};
                        markers.forEach(function (m) {
                            var id = String(m.id);
                            var mk;
                            if (dot) {
                                mk = L.circleMarker([m.lat, m.lng], {
                                    radius: dot.radius, weight: dot.weight, fillOpacity: dot.fill_opacity,
                                    fillColor: m.color, color: m.border || m.color
                                });
                            } else {
                                var icon = window.gaIcon(m.color);
                                mk = icon ? L.marker([m.lat, m.lng], {icon: icon}) : L.marker([m.lat, m.lng]);
                            }
                            if (m.tooltip) mk.bindTooltip(m.tooltip, {direction: 'top'});
                            mk.on('click', function () {
                                if (window.gaBridge) window.gaBridge.markerClicked(id);
                            });
                            mk.addTo(group);
                            byId[id] = mk;
                        });
                        group.addTo(window.map);
                        window.gaLayers[name] = {group: group, byId: byId, dot: !!dot};
                    };

                    window.gaSetMarkerColor = function (name, id, color, border) {
                        var layer = window.gaLayers[name];
                        if (!layer) return;
                        var mk = layer.byId[String(id)];
                        if (!mk) return;
                        if (layer.dot) {
                            mk.setStyle({fillColor: color, color: border || color});
                            mk.bringToFront();
                        } else {
                            var icon = window.gaIcon(color);
                            if (icon) mk.setIcon(icon);
                        }
                    };

                    // Box select: shift+drag draws a rectangle
                    var boxStart = null, boxRect = null;
                    function onMouseDown(e) {
                        if (!window.gaBoxSelectEnabled || !e.originalEvent.shiftKey) return;
                        boxStart = e.latlng;
                        window.map.dragging.disable();
                        boxRect = L.rectangle([boxStart, boxStart], {weight: 1, dashArray: '4'}).addTo(window.map);
                    }
                    function onMouseMove(e) {
                        if (boxStart && boxRect) boxRect.setBounds(L.latLngBounds(boxStart, e.latlng));
                    }
                    function onMouseUp(e) {
                        if (!boxStart) return;
                        var bounds = L.latLngBounds(boxStart, e.latlng);
                        window.map.removeLayer(boxRect);
                        boxRect = null;
                        boxStart = null;
                        window.map.dragging.enable();
                        var seen = {};
                        var ids = [];
                        Object.keys(window.gaLayers).forEach(function (name) {
                            var byId = window.gaLayers[name].byId;
                            Object.keys(byId).forEach(function (id) {
                                if (!seen[id] && bounds.contains(byId[id].getLatLng())) { seen[id] = true; ids.push(id); }
                            });
                        });
                        if (window.gaBridge) window.gaBridge.boxSelected(JSON.stringify(ids));
                    }

                    window.gaSetBoxSelect = function (enabled) {
                        window.gaBoxSelectEnabled = !!enabled;
                        if (!mapReady()) return;
                        if (enabled) window.map.boxZoom.disable(); else window.map.boxZoom.enable();
                    };

                    (function attach() {
                        if (!mapReady()) return setTimeout(attach, 100);
                        if (!window.gaBoxSelectEnabled) window.map.boxZoom.enable(); else window.map.boxZoom.disable();
                        window.map.on('moveend', onMoveEnd);
                        window.map.on('mousedown', onMouseDown);
                        window.map.on('mousemove', onMouseMove);
                        window.map.on('mouseup', onMouseUp);
                    })();
                })();
            </script>"""
        return (js.replace("__DEBOUNCE_MS__", str(int(self._bounds_debounce_ms)))
                  .replace("__OVERLAY_ORDER__", json.dumps(list(self.OVERLAY_ORDER)))
                  .replace("__OVERLAY_VISIBLE__", json.dumps(self._overlay_visible))
                  .replace("__CLUSTER_RADIUS_PX__", str(int(self.CLUSTER_RADIUS_PX)))
                  .replace("__CLUSTER_OFF_AT_ZOOM__", str(int(self.CLUSTER_OFF_AT_ZOOM)))
                  .replace("__CLUSTER_OVERLAP_PX__", str(int(self.CLUSTER_OVERLAP_PX))))

    # --------------------------------------------------------------------------------------------------------------
    # Internal: build and load HTML
    # --------------------------------------------------------------------------------------------------------------
    def _load_map(self):
        base_path = os.path.abspath(os.path.dirname(__file__))
        leaflet_dir = os.path.join(base_path, "../resources", "leaflet")
        leaflet_css = os.path.join(leaflet_dir, "leaflet.css")
        cluster_css = os.path.join(leaflet_dir, "MarkerCluster.css")
        cluster_js = os.path.join(leaflet_dir, "leaflet.markercluster.js")
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
            {f'<link rel="stylesheet" href="{http_url(cluster_css)}"/>' if os.path.exists(cluster_css) else ''}
            <style>
                .ga-cluster-icon {{ background: none; border: none; }}
                .ga-cluster {{ position: relative; }}
                .ga-count {{ position: absolute; top: -6px; right: -12px; min-width: 14px; padding: 0 4px;
                            background: #fff; color: #000; border: 1.5px solid #000; border-radius: 9px;
                            font: bold 11px sans-serif; text-align: center; line-height: 15px; }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script src="{http_url(leaflet_js)}"></script>
            {f'<script src="{http_url(cluster_js)}"></script>' if os.path.exists(cluster_js) else ''}
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
            {self._event_js()}
        </body>
        </html>"""

        # Write HTML to disk inside the served directory so it has an HTTP origin
        map_html_path = os.path.join(serve_root, "map.html")
        with open(map_html_path, "w", encoding="utf-8") as f:
            f.write(html)

        self.view.load(QUrl(f"http://127.0.0.1:{self._server_port}/map.html"))
