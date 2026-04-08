"""Sigil Configuration GUI — GTK4 + libadwaita.

A standalone application for managing Sigil gesture configuration,
recording training data, and triggering model training.

Launch with: sigil gui
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gdk, GLib, Gtk  # noqa: E402

from sigil.config import (  # noqa: E402
    CONFIG_FILE,
    DATA_DIR,
    MODELS_DIR,
    RECORDINGS_DIR,
    GestureMapping,
    SigilConfig,
    load_config,
    save_config,
)
from sigil.ui.drawing import GESTURE_EMOJIS  # noqa: E402

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MEDIAPIPE_POSES = [
    "Closed_Fist",
    "Open_Palm",
    "Pointing_Up",
    "Thumb_Up",
    "Thumb_Down",
    "Victory",
    "ILoveYou",
]
GRADUAL_FEATURES = [
    "two_hand_distance",
    "thumb_index_distance",
    "finger_curl_mean",
    "pointer_position",
    "finger_count",
]
HANDS = ["right", "left", "both"]
GESTURE_TYPES = ["instant", "gradual"]
DIRECTIONS = ["increase", "decrease"]
POSITIONS = ["bottom-right", "bottom-left", "top-right", "top-left"]
RESOLUTIONS = {"640×480": (640, 480), "1280×720": (1280, 720), "1920×1080": (1920, 1080)}

NAV_ITEMS = [
    ("gestures", "Gestures", "emoji-people-symbolic"),
    ("record", "Record", "camera-web-symbolic"),
    ("train", "Train", "system-run-symbolic"),
    ("settings", "Settings", "preferences-system-symbolic"),
    ("about", "About", "help-about-symbolic"),
]

_CSS_PATH = Path(__file__).parent / "app_styles.css"


def get_available_poses() -> list[str]:
    """Get all available poses for gesture mapping.

    Combines MediaPipe built-in poses with custom geometric model classes.
    """
    poses = list(MEDIAPIPE_POSES)

    # Check for geometric model classes
    geometric_path = MODELS_DIR / "custom_gesture.pkl"
    if geometric_path.exists():
        try:
            import joblib

            artifact = joblib.load(geometric_path)
            for cls_name in artifact.get("class_names", []):
                if cls_name.lower() != "none" and cls_name not in poses:
                    poses.append(cls_name)
        except Exception:
            pass

    # Also check recordings dir for any additional classes
    if RECORDINGS_DIR.exists():
        for d in sorted(RECORDINGS_DIR.iterdir()):
            if d.is_dir() and d.name.lower() != "none" and d.name not in poses:
                poses.append(d.name)
    return poses


def _gesture_emoji(g: GestureMapping) -> str:
    """Resolve emoji for a gesture mapping."""
    if g.type == "instant":
        pose = g.condition.get("pose", "")
        return GESTURE_EMOJIS.get(pose, "🤚")
    feat = g.condition.get("feature", "")
    return GESTURE_EMOJIS.get(feat, "🔄")


def _title(s: str) -> str:
    """Format snake_case to Title Case."""
    return s.replace("_", " ").title()


# ═════════════════════════════════════════════════════════════════════════════
#  Application
# ═════════════════════════════════════════════════════════════════════════════
class SigilApp(Adw.Application):
    """Top-level Adwaita application."""

    def __init__(self) -> None:
        super().__init__(application_id="dev.sigil.configurator")

    def do_activate(self) -> None:  # type: ignore[override]
        print("DEBUG: Application activating...")
        win = SigilWindow(application=self)
        print("DEBUG: Window created, presenting...")
        win.present()
        print("DEBUG: Window presented.")


# ═════════════════════════════════════════════════════════════════════════════
#  Main Window
# ═════════════════════════════════════════════════════════════════════════════
class SigilWindow(Adw.ApplicationWindow):
    """Configuration window with sidebar navigation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        print("DEBUG: Loading config...")
        self._cfg = load_config()
        print("DEBUG: Config loaded.")

        # Recording state
        self._camera_running = False
        self._camera_thread: threading.Thread | None = None
        self._current_frame: Any = None  # FrameResult
        self._active_tracker: Any = None
        self._active_recorder: Any = None
        self._record_preview: Gtk.DrawingArea | None = None
        self._record_status_label: Gtk.Label | None = None
        self._record_counter_label: Gtk.Label | None = None
        self._record_btn: Gtk.Button | None = None

        # Training state
        self._train_spinner: Gtk.Spinner | None = None
        self._train_status: Gtk.Label | None = None

        # State tracking for dynamic lists
        self._record_rows: list[Gtk.Widget] = []
        self._train_record_rows: list[Gtk.Widget] = []

        print("DEBUG: Loading CSS...")
        self._load_css()
        print("DEBUG: Building UI...")
        self._build_ui()
        print("DEBUG: UI Built.")

    def _load_css(self) -> None:
        if _CSS_PATH.exists():
            provider = Gtk.CssProvider()
            provider.load_from_path(str(_CSS_PATH))
            Gtk.StyleContext.add_provider_for_display(
                Gdk.Display.get_default(),
                provider,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
            )

        style_mgr = Adw.StyleManager.get_default()
        style_mgr.set_color_scheme(Adw.ColorScheme.FORCE_DARK)

    def _build_ui(self) -> None:
        self.set_default_size(920, 650)
        self.set_title("Sigil")

        # Toast overlay wrapping everything
        self._toast_overlay = Adw.ToastOverlay()
        self.set_content(self._toast_overlay)

        # Main split: sidebar + content
        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self._toast_overlay.set_child(main_box)

        main_box.append(self._build_sidebar())

        # Content side
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, hexpand=True)
        main_box.append(content_box)

        # Content header
        self._content_header = Adw.HeaderBar()
        self._content_header.set_show_end_title_buttons(True)
        self._content_header.set_show_start_title_buttons(False)
        self._header_title = Gtk.Label(label="Gestures")
        self._header_title.add_css_class("page-title")
        self._content_header.set_title_widget(self._header_title)

        # Launch Daemon Button
        self._daemon_proc = None
        self._daemon_btn = Gtk.Button(label="Start Overlay")
        self._daemon_btn.add_css_class("suggested-action")
        self._daemon_btn.set_tooltip_text("Launch Sigil Tracking Daemon")
        self._daemon_btn.connect("clicked", self._toggle_daemon)
        self._content_header.pack_end(self._daemon_btn)

        # Add Gesture Button (toggled visibility)
        self._add_gesture_btn = Gtk.Button(icon_name="list-add-symbolic")
        self._add_gesture_btn.set_tooltip_text("Add Gesture")
        self._add_gesture_btn.connect("clicked", lambda _: self._show_gesture_dialog())
        self._content_header.pack_end(self._add_gesture_btn)

        content_box.append(self._content_header)

        # Page stack
        self._stack = Gtk.Stack()
        self._stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self._stack.set_transition_duration(200)
        self._stack.set_hexpand(True)
        self._stack.set_vexpand(True)
        content_box.append(self._stack)

        # Build pages
        self._stack.add_named(self._build_gestures_page(), "gestures")
        self._stack.add_named(self._build_record_page(), "record")
        self._stack.add_named(self._build_train_page(), "train")
        self._stack.add_named(self._build_settings_page(), "settings")
        self._stack.add_named(self._build_about_page(), "about")

        # Select first page now that stack is built
        first = self._nav_list.get_row_at_index(0)
        if first:
            self._nav_list.select_row(first)

    def _toggle_daemon(self, btn: Gtk.Button) -> None:
        import subprocess

        if self._daemon_proc and self._daemon_proc.poll() is None:
            self._daemon_proc.terminate()
            self._daemon_proc = None
            btn.set_label("Start Overlay")
            btn.remove_css_class("destructive-action")
            btn.add_css_class("suggested-action")
            self._toast("Overlay stopped")
        else:
            # Check if camera is running in record page
            if getattr(self, "_camera_running", False):
                self._stop_camera()
                self._toast("Stopped camera preview for daemon")

            try:
                self._daemon_proc = subprocess.Popen(
                    ["sigil", "run"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                btn.set_label("Stop Overlay")
                btn.remove_css_class("suggested-action")
                btn.add_css_class("destructive-action")
                self._toast("Overlay started")
            except Exception as e:
                self._toast(f"Failed to start overlay: {e}")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    def _build_sidebar(self) -> Gtk.Widget:
        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        sidebar.set_size_request(220, -1)
        sidebar.add_css_class("sidebar-box")

        # Header (no window buttons, no title)
        sb_header = Adw.HeaderBar()
        sb_header.set_show_end_title_buttons(False)
        sb_header.set_show_start_title_buttons(True)
        sb_header.set_title_widget(Gtk.Label())  # empty
        sidebar.append(sb_header)

        # Brand
        brand = Gtk.Label(label="SIGIL")
        brand.set_halign(Gtk.Align.START)
        brand.add_css_class("sidebar-brand")
        sidebar.append(brand)

        sub = Gtk.Label(label="GESTURE CONTROL")
        sub.set_halign(Gtk.Align.START)
        sub.add_css_class("sidebar-subtitle")
        sidebar.append(sub)

        # Navigation
        self._nav_list = Gtk.ListBox()
        self._nav_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self._nav_list.add_css_class("nav-listbox")
        self._nav_list.connect("row-selected", self._on_nav_selected)

        for page_id, label, icon_name in NAV_ITEMS:
            row = Gtk.ListBoxRow()
            row.page_id = page_id  # type: ignore[attr-defined]
            box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
            icon = Gtk.Image.new_from_icon_name(icon_name)
            icon.add_css_class("nav-icon")
            lbl = Gtk.Label(label=label)
            lbl.add_css_class("nav-label")
            lbl.set_halign(Gtk.Align.START)
            box.append(icon)
            box.append(lbl)
            row.set_child(box)
            self._nav_list.append(row)

        sidebar.append(self._nav_list)
        return sidebar

    def _on_nav_selected(self, listbox: Gtk.ListBox, row: Gtk.ListBoxRow | None) -> None:
        if row is None:
            return
        page_id = row.page_id  # type: ignore[attr-defined]
        self._stack.set_visible_child_name(page_id)

        # Update header title
        for pid, label, _ in NAV_ITEMS:
            if pid == page_id:
                self._header_title.set_label(label)
                break

        # Add gesture button in header for gestures page
        self._update_header_actions(page_id)

        # Camera management
        if page_id == "record":
            self._start_camera()
        else:
            self._stop_camera()

    def _update_header_actions(self, page_id: str) -> None:
        self._add_gesture_btn.set_visible(page_id == "gestures")

    # ── Toast helper ─────────────────────────────────────────────────────────
    def _toast(self, msg: str, timeout: int = 2) -> None:
        t = Adw.Toast(title=msg, timeout=timeout)
        self._toast_overlay.add_toast(t)

    # ── Save helper ──────────────────────────────────────────────────────────
    def _save(self) -> None:
        save_config(self._cfg)
        logger.info("Config saved")

    # ═════════════════════════════════════════════════════════════════════════
    #  GESTURES PAGE
    # ═════════════════════════════════════════════════════════════════════════
    def _build_gestures_page(self) -> Gtk.Widget:
        self._gestures_container = Gtk.ScrolledWindow()
        self._gestures_container.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._refresh_gestures()
        return self._gestures_container

    def _refresh_gestures(self) -> None:
        page = Adw.PreferencesPage()

        if not self._cfg.gestures:
            # Empty state
            empty = Gtk.Box(
                orientation=Gtk.Orientation.VERTICAL,
                spacing=12,
                halign=Gtk.Align.CENTER,
                valign=Gtk.Align.CENTER,
                vexpand=True,
            )
            icon = Gtk.Label(label="🤚")
            icon.add_css_class("empty-icon")
            empty.append(icon)
            lbl = Gtk.Label(label="No gestures configured")
            lbl.add_css_class("empty-label")
            empty.append(lbl)
            hint = Gtk.Label(label='Click "+" to add your first gesture')
            hint.add_css_class("empty-label")
            empty.append(hint)
            self._gestures_container.set_child(empty)
            return

        # Group by hand
        groups: dict[str, list[GestureMapping]] = {"right": [], "left": [], "both": []}
        for g in self._cfg.gestures:
            groups.setdefault(g.hand, []).append(g)

        hand_labels = {"right": "Right Hand", "left": "Left Hand", "both": "Both Hands"}

        for hand in ("right", "left", "both"):
            gestures = groups.get(hand, [])
            if not gestures:
                continue

            group = Adw.PreferencesGroup(title=hand_labels[hand])
            for g in gestures:
                row = self._make_gesture_row(g)
                group.add(row)
            page.add(group)

        self._gestures_container.set_child(page)

    def _make_gesture_row(self, g: GestureMapping) -> Adw.ActionRow:
        emoji = _gesture_emoji(g)
        row = Adw.ActionRow(title=f"{emoji}  {_title(g.name)}")

        # Subtitle: type · cooldown
        parts = [g.type, f"{g.cooldown_ms}ms"]
        if g.type == "instant":
            parts.insert(0, g.condition.get("pose", "?"))
        row.set_subtitle(" · ".join(parts))

        # Hand badge
        badge = Gtk.Label(label=g.hand.upper())
        badge.add_css_class("hand-badge")
        if g.hand == "left":
            badge.add_css_class("hand-badge-left")
        elif g.hand == "both":
            badge.add_css_class("hand-badge-both")
        badge.set_valign(Gtk.Align.CENTER)
        row.add_prefix(badge)

        # Toggle switch
        switch = Gtk.Switch()
        switch.set_active(g.enabled)
        switch.set_valign(Gtk.Align.CENTER)
        switch.connect("state-set", self._on_gesture_toggle, g)
        row.add_suffix(switch)

        # Edit button
        edit_btn = Gtk.Button(icon_name="document-edit-symbolic")
        edit_btn.set_valign(Gtk.Align.CENTER)
        edit_btn.add_css_class("flat")
        edit_btn.set_tooltip_text("Edit")
        edit_btn.connect("clicked", lambda _, gest=g: self._show_gesture_dialog(gest))
        row.add_suffix(edit_btn)

        # Delete button
        del_btn = Gtk.Button(icon_name="user-trash-symbolic")
        del_btn.set_valign(Gtk.Align.CENTER)
        del_btn.add_css_class("flat")
        del_btn.add_css_class("delete-btn")
        del_btn.set_tooltip_text("Delete")
        del_btn.connect("clicked", lambda _, gest=g: self._delete_gesture(gest))
        row.add_suffix(del_btn)

        row.set_activatable(True)
        row.connect("activated", lambda _, gest=g: self._show_gesture_dialog(gest))
        return row

    def _on_gesture_toggle(self, switch: Gtk.Switch, state: bool, g: GestureMapping) -> bool:
        g.enabled = state
        self._save()
        self._toast(f"{'Enabled' if state else 'Disabled'} {_title(g.name)}")
        return False

    def _delete_gesture(self, g: GestureMapping) -> None:
        self._cfg.gestures.remove(g)
        self._save()
        self._refresh_gestures()
        self._toast(f"Deleted {_title(g.name)}")

    # ── Gesture Edit Dialog ──────────────────────────────────────────────────
    def _show_gesture_dialog(self, gesture: GestureMapping | None = None) -> None:
        is_new = gesture is None
        dlg_title = "New Gesture" if is_new else f"Edit {_title(gesture.name)}"  # type: ignore[union-attr]

        # Use a new window as modal dialog
        dialog = Gtk.Window(
            title=dlg_title,
            transient_for=self,
            modal=True,
            default_width=480,
            default_height=580,
            resizable=False,
        )

        # Content
        toolbar_view = Adw.ToolbarView()
        header = Adw.HeaderBar()
        toolbar_view.add_top_bar(header)

        # Cancel / Save buttons
        cancel_btn = Gtk.Button(label="Cancel")
        cancel_btn.connect("clicked", lambda _: dialog.close())
        header.pack_start(cancel_btn)

        save_btn = Gtk.Button(label="Save")
        save_btn.add_css_class("suggested-action")
        header.pack_end(save_btn)

        dialog.set_child(toolbar_view)

        # Form
        page = Adw.PreferencesPage()
        toolbar_view.set_content(page)

        # ── Basic info group ──
        basic_group = Adw.PreferencesGroup(title="Basic")

        name_entry = Adw.EntryRow(title="Name")
        name_entry.set_text(gesture.name if gesture else "")
        basic_group.add(name_entry)

        type_combo = Adw.ComboRow(title="Type", model=Gtk.StringList.new(GESTURE_TYPES))
        if gesture:
            try:
                type_combo.set_selected(GESTURE_TYPES.index(gesture.type))
            except ValueError:
                pass
        basic_group.add(type_combo)

        hand_combo = Adw.ComboRow(title="Hand", model=Gtk.StringList.new(HANDS))
        if gesture:
            try:
                hand_combo.set_selected(HANDS.index(gesture.hand))
            except ValueError:
                pass
        basic_group.add(hand_combo)

        page.add(basic_group)

        # ── Instant condition group ──
        instant_group = Adw.PreferencesGroup(title="Instant Condition")

        poses = get_available_poses()
        pose_combo = Adw.ComboRow(title="Pose", model=Gtk.StringList.new(poses))
        if gesture and gesture.type == "instant":
            pose = gesture.condition.get("pose", "")
            if pose in poses:
                pose_combo.set_selected(poses.index(pose))
        instant_group.add(pose_combo)
        page.add(instant_group)

        # ── Gradual condition group ──
        gradual_group = Adw.PreferencesGroup(title="Gradual Condition")

        feature_combo = Adw.ComboRow(title="Feature", model=Gtk.StringList.new(GRADUAL_FEATURES))
        if gesture and gesture.type == "gradual":
            feat = gesture.condition.get("feature", "")
            if feat in GRADUAL_FEATURES:
                feature_combo.set_selected(GRADUAL_FEATURES.index(feat))
        gradual_group.add(feature_combo)

        delta_adj = Gtk.Adjustment(
            value=gesture.condition.get("min_delta", 0.05) if gesture else 0.05,
            lower=0.01,
            upper=1.0,
            step_increment=0.01,
        )
        delta_spin = Adw.SpinRow(title="Min Delta", adjustment=delta_adj, digits=2)
        gradual_group.add(delta_spin)

        dir_combo = Adw.ComboRow(title="Direction", model=Gtk.StringList.new(DIRECTIONS))
        if gesture and gesture.type == "gradual":
            d = gesture.condition.get("direction", "increase")
            if d in DIRECTIONS:
                dir_combo.set_selected(DIRECTIONS.index(d))
        gradual_group.add(dir_combo)
        page.add(gradual_group)

        # ── Action group ──
        action_group = Adw.PreferencesGroup(title="Action")

        action_entry = Adw.EntryRow(title="Command")
        action_entry.set_text(gesture.action if gesture else "hyprctl dispatch ")
        action_group.add(action_entry)

        cooldown_adj = Gtk.Adjustment(
            value=gesture.cooldown_ms if gesture else 500,
            lower=0,
            upper=10000,
            step_increment=50,
        )
        cooldown_spin = Adw.SpinRow(title="Cooldown (ms)", adjustment=cooldown_adj, digits=0)
        action_group.add(cooldown_spin)

        enabled_switch = Adw.SwitchRow(title="Enabled")
        enabled_switch.set_active(gesture.enabled if gesture else True)
        action_group.add(enabled_switch)
        page.add(action_group)

        # ── Show/hide condition groups based on type ──
        def update_visibility(*_args: Any) -> None:
            sel = GESTURE_TYPES[type_combo.get_selected()]
            instant_group.set_visible(sel == "instant")
            gradual_group.set_visible(sel == "gradual")
            # Force "both" for gradual two_hand_distance
            if sel == "gradual":
                feat = GRADUAL_FEATURES[feature_combo.get_selected()]
                if feat == "two_hand_distance":
                    hand_combo.set_selected(HANDS.index("both"))

        type_combo.connect("notify::selected", update_visibility)
        update_visibility()

        # ── Save handler ──
        def on_save(_btn: Gtk.Button) -> None:
            name = name_entry.get_text().strip()
            if not name:
                self._toast("Name is required")
                return

            # Sanitise name
            name = name.lower().replace(" ", "_")

            gtype = GESTURE_TYPES[type_combo.get_selected()]
            hand = HANDS[hand_combo.get_selected()]
            action = action_entry.get_text().strip()
            cooldown = int(cooldown_adj.get_value())
            enabled = enabled_switch.get_active()

            condition: dict[str, Any] = {}
            if gtype == "instant":
                condition["pose"] = poses[pose_combo.get_selected()]
            elif gtype == "gradual":
                condition["feature"] = GRADUAL_FEATURES[feature_combo.get_selected()]
                condition["min_delta"] = delta_adj.get_value()
                condition["direction"] = DIRECTIONS[dir_combo.get_selected()]

            if is_new:
                new_g = GestureMapping(
                    name=name,
                    type=gtype,
                    hand=hand,
                    condition=condition,
                    action=action,
                    cooldown_ms=cooldown,
                    enabled=enabled,
                )
                self._cfg.gestures.append(new_g)
            else:
                assert gesture is not None
                gesture.name = name
                gesture.type = gtype
                gesture.hand = hand
                gesture.condition = condition
                gesture.action = action
                gesture.cooldown_ms = cooldown
                gesture.enabled = enabled

            self._save()
            self._refresh_gestures()
            dialog.close()
            self._toast(f"{'Created' if is_new else 'Updated'} {_title(name)}")

        save_btn.connect("clicked", on_save)
        dialog.present()

    # ═════════════════════════════════════════════════════════════════════════
    #  RECORD PAGE
    # ═════════════════════════════════════════════════════════════════════════
    def _build_record_page(self) -> Gtk.Widget:
        scroll = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
        )
        outer = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=20,
            margin_top=24,
            margin_bottom=24,
            margin_start=32,
            margin_end=32,
        )
        scroll.set_child(outer)

        # Camera preview
        preview_frame = Gtk.Frame()
        preview_frame.add_css_class("camera-preview")
        self._record_preview = Gtk.DrawingArea()
        self._record_preview.set_size_request(400, 300)
        self._record_preview.set_content_width(400)
        self._record_preview.set_content_height(300)
        self._record_preview.set_halign(Gtk.Align.CENTER)
        self._record_preview.set_draw_func(self._draw_camera)
        preview_frame.set_child(self._record_preview)
        outer.append(preview_frame)

        # Status
        status_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=16,
            halign=Gtk.Align.CENTER,
        )
        self._record_status_label = Gtk.Label(label="Camera offline")
        self._record_status_label.add_css_class("page-subtitle")
        status_box.append(self._record_status_label)

        self._record_counter_label = Gtk.Label(label="")
        self._record_counter_label.add_css_class("sample-counter")
        status_box.append(self._record_counter_label)
        outer.append(status_box)

        # Controls
        controls = Adw.PreferencesGroup(title="Recording Controls")

        self._rec_name_entry = Adw.EntryRow(title="Gesture Name")
        self._rec_name_entry.set_text("my_gesture")
        controls.add(self._rec_name_entry)

        self._rec_mode_combo = Adw.ComboRow(
            title="Mode",
            model=Gtk.StringList.new(["instant", "gradual"]),
        )
        controls.add(self._rec_mode_combo)

        self._rec_hand_combo = Adw.ComboRow(
            title="Hand",
            model=Gtk.StringList.new(HANDS),
        )
        controls.add(self._rec_hand_combo)

        self._rec_target_adj = Gtk.Adjustment(
            value=100,
            lower=10,
            upper=500,
            step_increment=10,
        )
        target_spin = Adw.SpinRow(
            title="Target Samples",
            adjustment=self._rec_target_adj,
            digits=0,
        )
        controls.add(target_spin)
        outer.append(controls)

        # Buttons
        btn_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            halign=Gtk.Align.CENTER,
            margin_top=8,
        )
        self._record_btn = Gtk.Button(label="⏺  Start Recording")
        self._record_btn.add_css_class("record-start-btn")
        self._record_btn.connect("clicked", self._on_record_toggle)
        btn_box.append(self._record_btn)
        outer.append(btn_box)

        # Existing recordings
        self._recordings_group = Adw.PreferencesGroup(title="Recorded Classes")
        outer.append(self._recordings_group)
        self._refresh_recordings_list()

        return scroll

    def _refresh_recordings_list(self) -> None:
        # Clear old rows
        for row in self._record_rows:
            self._recordings_group.remove(row)
        self._record_rows.clear()

        # Rebuild
        rec_dir = RECORDINGS_DIR
        if not rec_dir.exists():
            return

        found = False
        for cls_dir in sorted(rec_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            jpgs = len(list(cls_dir.glob("*.jpg")))
            jsons = len(list(cls_dir.glob("*.json")))
            total = jpgs + jsons
            if total == 0:
                continue
            found = True
            row = Adw.ActionRow(
                title=_title(cls_dir.name),
                subtitle=f"{total} samples ({jpgs} images, {jsons} features)",
            )
            row.add_prefix(Gtk.Label(label="📁"))
            self._recordings_group.add(row)
            self._record_rows.append(row)

        if not found:
            row = Adw.ActionRow(
                title="No recordings yet",
                subtitle="Record gesture samples to train custom models",
            )
            self._recordings_group.add(row)
            self._record_rows.append(row)

    def _draw_camera(self, area: Gtk.DrawingArea, cr: Any, width: int, height: int) -> None:
        """Draw camera preview frame."""
        import cairo

        # Background
        cr.set_source_rgba(0.067, 0.067, 0.106, 1.0)
        cr.rectangle(0, 0, width, height)
        cr.fill()

        frame = self._current_frame
        if frame is None or frame.frame is None:
            # Placeholder text
            cr.set_source_rgba(0.65, 0.68, 0.78, 0.4)
            cr.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            cr.set_font_size(14)
            txt = "Navigate to Record to start camera"
            ext = cr.text_extents(txt)
            cr.move_to((width - ext.width) / 2, (height + ext.height) / 2)
            cr.show_text(txt)
            return

        # Draw the BGR frame
        import cv2

        # cairo.FORMAT_ARGB32 expects memory stored as BGRA on little-endian platforms.
        bgra = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2BGRA)
        h, w = bgra.shape[:2]

        # Scale to fit
        scale = min(width / w, height / h)
        sw, sh = int(w * scale), int(h * scale)
        bgra = cv2.resize(bgra, (sw, sh))

        surface = cairo.ImageSurface.create_for_data(
            bytearray(bgra.tobytes()), cairo.FORMAT_ARGB32, sw, sh
        )
        # Center
        ox = (width - sw) / 2
        oy = (height - sh) / 2
        cr.translate(ox, oy)
        cr.set_source_surface(surface, 0, 0)
        cr.paint()

        # Draw landmarks if hands detected
        if frame.hands:
            from sigil.ui.drawing import draw_hand

            for hand in frame.hands:
                draw_hand(cr, hand, sw, sh, confidence=hand.score)

        cr.translate(-ox, -oy)

        # Recording indicator
        if self._active_recorder and self._active_recorder.is_active:
            cr.set_source_rgba(0.95, 0.55, 0.66, 0.9)
            cr.arc(20, 20, 8, 0, 2 * 3.14159)
            cr.fill()

    def _start_camera(self) -> None:
        if self._camera_running:
            return

        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["GDK_BACKEND"] = "x11"
        os.environ["EGL_DRIVER"] = "sw"

        self._camera_running = True
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()
        if self._record_status_label:
            self._record_status_label.set_label("Camera starting…")

    def _stop_camera(self) -> None:
        if not self._camera_running:
            return
        self._camera_running = False
        if self._camera_thread:
            self._camera_thread.join(timeout=3)
            self._camera_thread = None
        self._current_frame = None
        if self._record_status_label:
            self._record_status_label.set_label("Camera offline")

    def _camera_loop(self) -> None:
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["GDK_BACKEND"] = "x11"
        os.environ["EGL_DRIVER"] = "sw"

        # Preload TensorFlow stack to fix EGL conflicts
        import mediapipe
        import tensorflow

        from sigil.tracker import Tracker

        try:
            tracker = Tracker(self._cfg.tracking)
            tracker.open()
            self._active_tracker = tracker
            GLib.idle_add(
                lambda: (
                    self._record_status_label.set_label("Camera active")  # type: ignore[union-attr]
                    if self._record_status_label
                    else None
                )
            )
        except Exception as e:
            GLib.idle_add(
                lambda e=e: (
                    self._record_status_label.set_label(f"Camera error: {e}")  # type: ignore[union-attr]
                    if self._record_status_label
                    else None
                )
            )
            return

        target_delay = 1.0 / max(1, self._cfg.tracking.target_fps)
        while self._camera_running:
            t0 = time.monotonic()
            frame = tracker.read()
            if frame is not None:
                self._current_frame = frame

                # If recording, capture
                if self._active_recorder and self._active_recorder.is_active:
                    self._active_recorder.capture(frame)
                    sess = self._active_recorder.session
                    GLib.idle_add(self._update_record_counter, sess.sample_count, sess.target_count)

                GLib.idle_add(self._record_preview.queue_draw)  # type: ignore[union-attr]

            elapsed = time.monotonic() - t0
            sleep_time = target_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        tracker.close()
        self._active_tracker = None

    def _update_record_counter(self, count: int, target: int) -> None:
        if self._record_counter_label:
            self._record_counter_label.set_label(f"{count}/{target}")
        if count >= target and self._active_recorder and self._active_recorder.is_active:
            self._stop_recording()
            self._toast(f"Recording complete! {count} samples saved.")

    def _on_record_toggle(self, btn: Gtk.Button) -> None:
        if self._active_recorder and self._active_recorder.is_active:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        if not self._camera_running:
            self._toast("Camera not active — switch to Record page first")
            return

        from sigil.recorder import Recorder, RecordMode

        name = self._rec_name_entry.get_text().strip().lower().replace(" ", "_")
        if not name:
            self._toast("Enter a gesture name")
            return

        mode_str = ["instant", "gradual"][self._rec_mode_combo.get_selected()]
        mode = RecordMode.INSTANT if mode_str == "instant" else RecordMode.GRADUAL
        hand = HANDS[self._rec_hand_combo.get_selected()]
        target = int(self._rec_target_adj.get_value())

        recorder = Recorder(self._cfg.recording)
        recorder.start(name, mode, hand=hand, target_count=target)
        self._active_recorder = recorder

        if self._record_btn:
            self._record_btn.set_label("⏹  Stop Recording")
            self._record_btn.remove_css_class("record-start-btn")
            self._record_btn.add_css_class("record-stop-btn")

        if self._record_status_label:
            self._record_status_label.set_label(f"Recording '{_title(name)}'…")

        self._toast(f"Recording started: {_title(name)} ({mode_str})")

    def _stop_recording(self) -> None:
        if self._active_recorder:
            count = self._active_recorder.session.sample_count
            self._active_recorder.stop()
            self._active_recorder = None

        if self._record_btn:
            self._record_btn.set_label("⏺  Start Recording")
            self._record_btn.remove_css_class("record-stop-btn")
            self._record_btn.add_css_class("record-start-btn")

        if self._record_status_label:
            self._record_status_label.set_label("Camera active")

        if self._record_counter_label:
            self._record_counter_label.set_label("")

        self._refresh_recordings_list()
        self._refresh_train_data()
        self._toast("Recording stopped")

    # ═════════════════════════════════════════════════════════════════════════
    #  TRAIN PAGE
    # ═════════════════════════════════════════════════════════════════════════
    def _build_train_page(self) -> Gtk.Widget:
        scroll = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
        )
        page = Adw.PreferencesPage()
        scroll.set_child(page)

        # ── Model Status ──
        status_group = Adw.PreferencesGroup(title="Model Status")

        models = [
            ("gesture_recognizer.task", "Gesture Recognizer", "Built-in MediaPipe model"),
            ("hand_landmarker.task", "Hand Landmarker", "Built-in tracking model"),
            ("custom_gesture.pkl", "Geometric Classifier", "Random Forest on 96-dim landmarks"),
        ]

        for fname, title, desc in models:
            fpath = MODELS_DIR / fname
            row = Adw.ActionRow(title=title)
            if fpath.exists():
                size_kb = fpath.stat().st_size / 1024
                mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(fpath.stat().st_mtime))
                row.set_subtitle(f"{desc} — {size_kb:.0f} KB — {mtime}")
                check = Gtk.Image.new_from_icon_name("emblem-ok-symbolic")
                check.add_css_class("model-status-exists")
                row.add_suffix(check)
            else:
                row.set_subtitle(f"{desc} — not found")
                missing = Gtk.Label(label="missing")
                missing.add_css_class("model-status-missing")
                row.add_suffix(missing)
            status_group.add(row)

        page.add(status_group)

        # ── Recordings Summary ──
        self._train_recordings_group = Adw.PreferencesGroup(
            title="Training Data",
            description="Geometric training requires a 'None' class with background/noise samples. "
            "Ensure you have at least one custom gesture PLUS the 'None' class.",
        )
        page.add(self._train_recordings_group)
        self._refresh_train_data()

        # ── Train Buttons ──
        train_group = Adw.PreferencesGroup(title="Train")

        # Status row
        self._train_spinner = Gtk.Spinner()

        status_container = Adw.ActionRow(title="Status")
        status_container.add_suffix(self._train_spinner)
        status_container.set_subtitle("Ready to train")
        self._train_status_row = status_container
        train_group.add(status_container)

        # Buttons in a box
        btn_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            margin_start=16,
            margin_end=16,
            margin_top=12,
            margin_bottom=12,
            halign=Gtk.Align.CENTER,
        )

        instant_btn = Gtk.Button(label="Train Geometric")
        instant_btn.add_css_class("train-btn")
        instant_btn.set_tooltip_text(
            "Random Forest on 96-dim geometric features → custom_gesture.pkl"
        )
        instant_btn.connect("clicked", lambda _: self._run_training("instant"))
        btn_box.append(instant_btn)

        dynamic_btn = Gtk.Button(label="Train Sequential")
        dynamic_btn.add_css_class("train-btn")
        dynamic_btn.set_tooltip_text("Scikit-learn MLP + KNN from sequential landmark features")
        dynamic_btn.connect("clicked", lambda _: self._run_training("dynamic"))
        btn_box.append(dynamic_btn)

        all_btn = Gtk.Button(label="Train All")
        all_btn.add_css_class("suggested-action")
        all_btn.set_tooltip_text("Retrain all models from current recordings")
        all_btn.connect("clicked", lambda _: self._run_training("all"))
        btn_box.append(all_btn)

        # Wrap buttons in a row-like container
        btn_wrapper = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        btn_wrapper.append(btn_box)
        train_group.add(btn_wrapper)

        page.add(train_group)
        return scroll

    def _refresh_train_data(self) -> None:
        for row in self._train_record_rows:
            self._train_recordings_group.remove(row)
        self._train_record_rows.clear()

        rec_dir = RECORDINGS_DIR
        if rec_dir.exists():
            class_count = 0
            has_none = False
            for cls_dir in sorted(rec_dir.iterdir()):
                if not cls_dir.is_dir():
                    continue
                n = len(list(cls_dir.glob("*.jpg"))) + len(list(cls_dir.glob("*.json")))
                if n > 0:
                    class_count += 1
                    is_none = cls_dir.name.lower() == "none"
                    if is_none:
                        has_none = True

                    row = Adw.ActionRow(
                        title=_title(cls_dir.name),
                        subtitle=f"{n} samples {'(Background Class)' if is_none else ''}",
                    )
                    icon = "emblem-important-symbolic" if is_none else "folder-symbolic"
                    row.add_prefix(Gtk.Image.new_from_icon_name(icon))
                    if is_none:
                        row.add_css_class("none-class-row")

                    self._train_recordings_group.add(row)
                    self._train_record_rows.append(row)

            if not has_none and class_count > 0:
                row = Adw.ActionRow(
                    title="Missing 'None' class",
                    subtitle="Record some background samples to enable MediaPipe training",
                )
                row.add_prefix(Gtk.Image.new_from_icon_name("dialog-warning-symbolic"))
                row.add_css_class("error")
                self._train_recordings_group.add(row)
                self._train_record_rows.append(row)

            if class_count == 0:
                row = Adw.ActionRow(
                    title="No training data",
                    subtitle="Record gesture samples first",
                )
                self._train_recordings_group.add(row)
                self._train_record_rows.append(row)
        else:
            row = Adw.ActionRow(
                title="No training data",
                subtitle="Record gesture samples first",
            )
            self._train_recordings_group.add(row)
            self._train_record_rows.append(row)

    def _run_training(self, train_type: str) -> None:
        self._train_spinner.start()  # type: ignore[union-attr]
        self._train_status_row.set_subtitle(f"Training {train_type}…")  # type: ignore[union-attr]
        self._toast(f"Training {train_type} started…")

        def worker() -> None:
            try:
                from sigil.trainer import train_instant, retrain

                if train_type == "instant":
                    train_instant()
                elif train_type == "all":
                    retrain()
                elif train_type == "dynamic":
                    # Sequential/dynamic training not yet implemented
                    raise NotImplementedError("Sequential model training is not yet implemented")
                GLib.idle_add(self._on_training_done, train_type, None)  # type: ignore[name-defined]
            except Exception as e:
                GLib.idle_add(self._on_training_done, train_type, str(e))  # type: ignore[name-defined]

        threading.Thread(target=worker, daemon=True).start()

    def _on_training_done(self, train_type: str, error: str | None) -> None:
        self._train_spinner.stop()  # type: ignore[union-attr]
        if error:
            self._train_status_row.set_subtitle(f"Failed: {error}")  # type: ignore[union-attr]
            self._toast(f"Training failed: {error}")
        else:
            self._train_status_row.set_subtitle(f"✅ {_title(train_type)} training complete")  # type: ignore[union-attr]
            self._toast(f"Training complete: {train_type}")

            if train_type in ("instant", "all"):
                model_path = MODELS_DIR / "custom_gesture.pkl"
                if model_path.exists():
                    # Geometric model is consumed by GeometricClassifier directly.
                    # Do not write it into instant_model_path (expects MediaPipe .task).
                    current_instant_model = self._cfg.tracking.instant_model_path or ""
                    if Path(current_instant_model).name == "custom_gesture.pkl":
                        self._cfg.tracking.instant_model_path = None
                        self._save()
                        if hasattr(self, "_model_entry"):
                            self._model_entry.set_text("")

                    self._toast(
                        "Custom geometric model updated. Overlay now uses MediaPipe defaults + geometric custom gestures.",
                        timeout=3,
                    )

    # ═════════════════════════════════════════════════════════════════════════
    #  SETTINGS PAGE
    # ═════════════════════════════════════════════════════════════════════════
    def _build_settings_page(self) -> Gtk.Widget:
        scroll = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
        )
        page = Adw.PreferencesPage()
        scroll.set_child(page)
        cfg = self._cfg

        # ── Tracking ──
        tracking = Adw.PreferencesGroup(
            title="Tracking",
            description="Camera and hand detection settings",
        )

        cam_adj = Gtk.Adjustment(
            value=cfg.tracking.camera_id,
            lower=0,
            upper=10,
            step_increment=1,
        )
        cam_spin = Adw.SpinRow(
            title="Camera ID",
            subtitle="v4l2 device index",
            adjustment=cam_adj,
            digits=0,
        )
        cam_spin.connect(
            "notify::value",
            lambda s, _: setattr(cfg.tracking, "camera_id", int(s.get_value())) or self._save(),
        )
        tracking.add(cam_spin)

        res_options = list(RESOLUTIONS.keys())
        res_combo = Adw.ComboRow(
            title="Resolution",
            model=Gtk.StringList.new(res_options),
        )
        current_res = f"{cfg.tracking.frame_width}×{cfg.tracking.frame_height}"
        if current_res in res_options:
            res_combo.set_selected(res_options.index(current_res))

        def on_res_changed(combo: Adw.ComboRow, _p: Any) -> None:
            key = res_options[combo.get_selected()]
            w, h = RESOLUTIONS[key]
            cfg.tracking.frame_width = w
            cfg.tracking.frame_height = h
            self._save()

        res_combo.connect("notify::selected", on_res_changed)
        tracking.add(res_combo)

        fps_adj = Gtk.Adjustment(
            value=cfg.tracking.target_fps,
            lower=5,
            upper=30,
            step_increment=1,
        )
        fps_spin = Adw.SpinRow(
            title="Target FPS",
            subtitle="Frame rate for tracking loop",
            adjustment=fps_adj,
            digits=0,
        )
        fps_spin.connect(
            "notify::value",
            lambda s, _: setattr(cfg.tracking, "target_fps", int(s.get_value())) or self._save(),
        )
        tracking.add(fps_spin)

        det_adj = Gtk.Adjustment(
            value=cfg.tracking.min_detection_confidence,
            lower=0.1,
            upper=1.0,
            step_increment=0.05,
        )
        det_spin = Adw.SpinRow(
            title="Detection Confidence",
            subtitle="Minimum to detect a new hand",
            adjustment=det_adj,
            digits=2,
        )
        det_spin.connect(
            "notify::value",
            lambda s, _: (
                setattr(cfg.tracking, "min_detection_confidence", round(s.get_value(), 2))
                or self._save()
            ),
        )
        tracking.add(det_spin)

        trk_adj = Gtk.Adjustment(
            value=cfg.tracking.min_tracking_confidence,
            lower=0.1,
            upper=1.0,
            step_increment=0.05,
        )
        trk_spin = Adw.SpinRow(
            title="Tracking Confidence",
            subtitle="Minimum to maintain tracked hand",
            adjustment=trk_adj,
            digits=2,
        )
        trk_spin.connect(
            "notify::value",
            lambda s, _: (
                setattr(cfg.tracking, "min_tracking_confidence", round(s.get_value(), 2))
                or self._save()
            ),
        )
        tracking.add(trk_spin)

        zoom_adj = Gtk.Adjustment(
            value=cfg.tracking.zoom,
            lower=1.0,
            upper=3.0,
            step_increment=0.1,
        )
        zoom_spin = Adw.SpinRow(
            title="Digital Zoom",
            subtitle="1.0 = no zoom, higher = crop center",
            adjustment=zoom_adj,
            digits=1,
        )
        zoom_spin.connect(
            "notify::value",
            lambda s, _: setattr(cfg.tracking, "zoom", round(s.get_value(), 1)) or self._save(),
        )
        tracking.add(zoom_spin)

        self._model_entry = Adw.EntryRow(
            title="MediaPipe Instant .task Model Path",
        )
        self._model_entry.set_tooltip_text(
            "Optional MediaPipe .task path. Leave empty for built-in 7 default poses. "
            "Do not set to custom_gesture.pkl."
        )
        self._model_entry.set_text(cfg.tracking.instant_model_path or "")

        def on_model_changed(entry: Adw.EntryRow) -> None:
            val = entry.get_text().strip()
            cfg.tracking.instant_model_path = val if val else None
            self._save()

        self._model_entry.connect("changed", on_model_changed)
        tracking.add(self._model_entry)

        page.add(tracking)

        # ── Execution ──
        execution = Adw.PreferencesGroup(
            title="Execution",
            description="Gesture confirmation and debounce",
        )

        cf_adj = Gtk.Adjustment(
            value=cfg.execution.confirm_frames,
            lower=1,
            upper=10,
            step_increment=1,
        )
        cf_spin = Adw.SpinRow(
            title="Confirm Frames",
            subtitle="Consecutive frames needed before firing",
            adjustment=cf_adj,
            digits=0,
        )
        cf_spin.connect(
            "notify::value",
            lambda s, _: (
                setattr(cfg.execution, "confirm_frames", int(s.get_value())) or self._save()
            ),
        )
        execution.add(cf_spin)

        bl_adj = Gtk.Adjustment(
            value=cfg.execution.blanking_ms,
            lower=0,
            upper=5000,
            step_increment=100,
        )
        bl_spin = Adw.SpinRow(
            title="Blanking (ms)",
            subtitle="Global suppression after an action fires",
            adjustment=bl_adj,
            digits=0,
        )
        bl_spin.connect(
            "notify::value",
            lambda s, _: setattr(cfg.execution, "blanking_ms", int(s.get_value())) or self._save(),
        )
        execution.add(bl_spin)

        ct_adj = Gtk.Adjustment(
            value=cfg.execution.confidence_threshold,
            lower=0.1,
            upper=1.0,
            step_increment=0.05,
        )
        ct_spin = Adw.SpinRow(
            title="Confidence Threshold",
            subtitle="Minimum score to count toward confirmation",
            adjustment=ct_adj,
            digits=2,
        )
        ct_spin.connect(
            "notify::value",
            lambda s, _: (
                setattr(cfg.execution, "confidence_threshold", round(s.get_value(), 2))
                or self._save()
            ),
        )
        execution.add(ct_spin)

        page.add(execution)

        # ── Overlay ──
        overlay = Adw.PreferencesGroup(
            title="Overlay",
            description="On-screen gesture overlay appearance",
        )

        pos_combo = Adw.ComboRow(
            title="Position",
            model=Gtk.StringList.new(POSITIONS),
        )
        if cfg.overlay_settings.position in POSITIONS:
            pos_combo.set_selected(POSITIONS.index(cfg.overlay_settings.position))
        pos_combo.connect(
            "notify::selected",
            lambda c, _: (
                setattr(cfg.overlay_settings, "position", POSITIONS[c.get_selected()])
                or self._save()
            ),
        )
        overlay.add(pos_combo)

        cam_switch = Adw.SwitchRow(
            title="Camera Preview",
            subtitle="Show live camera feed in overlay",
        )
        cam_switch.set_active(cfg.overlay_settings.camera_preview)
        cam_switch.connect(
            "notify::active",
            lambda s, _: (
                setattr(cfg.overlay_settings, "camera_preview", s.get_active()) or self._save()
            ),
        )
        overlay.add(cam_switch)

        opa_adj = Gtk.Adjustment(
            value=cfg.overlay_settings.opacity,
            lower=0.1,
            upper=1.0,
            step_increment=0.1,
        )
        opa_spin = Adw.SpinRow(
            title="Opacity",
            subtitle="Overlay window transparency",
            adjustment=opa_adj,
            digits=1,
        )
        opa_spin.connect(
            "notify::value",
            lambda s, _: (
                setattr(cfg.overlay_settings, "opacity", round(s.get_value(), 1)) or self._save()
            ),
        )
        overlay.add(opa_spin)

        margin_adj = Gtk.Adjustment(
            value=cfg.overlay_settings.margin,
            lower=0,
            upper=64,
            step_increment=4,
        )
        margin_spin = Adw.SpinRow(
            title="Margin (px)",
            subtitle="Distance from screen edge",
            adjustment=margin_adj,
            digits=0,
        )
        margin_spin.connect(
            "notify::value",
            lambda s, _: (
                setattr(cfg.overlay_settings, "margin", int(s.get_value())) or self._save()
            ),
        )
        overlay.add(margin_spin)

        page.add(overlay)

        # ── Recording config ──
        recording = Adw.PreferencesGroup(
            title="Recording",
            description="Default recording parameters",
        )

        spc_adj = Gtk.Adjustment(
            value=cfg.recording.samples_per_class,
            lower=10,
            upper=500,
            step_increment=10,
        )
        spc_spin = Adw.SpinRow(
            title="Samples Per Class",
            subtitle="Default target when recording",
            adjustment=spc_adj,
            digits=0,
        )
        spc_spin.connect(
            "notify::value",
            lambda s, _: (
                setattr(cfg.recording, "samples_per_class", int(s.get_value())) or self._save()
            ),
        )
        recording.add(spc_spin)

        hotkey_entry = Adw.EntryRow(title="Recording Hotkey")
        hotkey_entry.set_text(cfg.recording.hotkey)
        hotkey_entry.connect(
            "changed",
            lambda e: setattr(cfg.recording, "hotkey", e.get_text()) or self._save(),
        )
        recording.add(hotkey_entry)

        page.add(recording)

        # ── Danger zone ──
        danger = Adw.PreferencesGroup(
            title="Advanced",
            description="Reset and maintenance",
        )

        reset_row = Adw.ActionRow(
            title="Reset to Defaults",
            subtitle="Restore all settings to factory defaults",
        )
        reset_btn = Gtk.Button(label="Reset")
        reset_btn.add_css_class("destructive-action")
        reset_btn.set_valign(Gtk.Align.CENTER)
        reset_btn.connect("clicked", self._on_reset_config)
        reset_row.add_suffix(reset_btn)
        danger.add(reset_row)

        page.add(danger)
        return scroll

    def _on_reset_config(self, _btn: Gtk.Button) -> None:
        self._cfg = SigilConfig()
        self._save()
        self._toast("Config reset to defaults — restart the app to see changes")

    # ═════════════════════════════════════════════════════════════════════════
    #  ABOUT PAGE
    # ═════════════════════════════════════════════════════════════════════════
    def _build_about_page(self) -> Gtk.Widget:
        from sigil import __version__

        box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=16,
            halign=Gtk.Align.CENTER,
            valign=Gtk.Align.CENTER,
            vexpand=True,
            margin_top=48,
        )

        title = Gtk.Label(label="SIGIL")
        title.add_css_class("about-title")
        box.append(title)

        ver = Gtk.Label(label=f"v{__version__}")
        ver.add_css_class("about-version")
        box.append(ver)

        desc = Gtk.Label(
            label="Gesture Control Framework for Hyprland",
            wrap=True,
            max_width_chars=50,
        )
        desc.add_css_class("page-subtitle")
        box.append(desc)

        box.append(Gtk.Separator(margin_top=16, margin_bottom=16))

        # Data paths
        paths_group = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=8,
            halign=Gtk.Align.CENTER,
        )

        path_items = [
            ("Config", str(CONFIG_FILE)),
            ("Models", str(MODELS_DIR)),
            ("Recordings", str(RECORDINGS_DIR)),
            ("Data", str(DATA_DIR)),
        ]

        for label, path in path_items:
            row = Gtk.Box(
                orientation=Gtk.Orientation.HORIZONTAL,
                spacing=12,
            )
            lbl = Gtk.Label(label=f"{label}:")
            lbl.set_halign(Gtk.Align.END)
            lbl.set_size_request(100, -1)
            lbl.add_css_class("page-subtitle")
            row.append(lbl)

            path_lbl = Gtk.Label(label=path)
            path_lbl.set_halign(Gtk.Align.START)
            path_lbl.add_css_class("about-path-label")
            path_lbl.set_selectable(True)
            row.append(path_lbl)

            paths_group.append(row)

        box.append(paths_group)

        # Open config button
        open_btn = Gtk.Button(label="Open Config Directory")
        open_btn.add_css_class("flat")
        open_btn.set_margin_top(24)
        open_btn.connect(
            "clicked",
            lambda _: os.system(f"xdg-open {CONFIG_FILE.parent}"),
        )
        box.append(open_btn)

        return box

    # ── Cleanup ──────────────────────────────────────────────────────────────
    def do_close_request(self) -> bool:  # type: ignore[override]
        self._stop_camera()
        if self._active_recorder and self._active_recorder.is_active:
            self._active_recorder.stop()
        return False


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════
def run_gui() -> None:
    """Launch the Sigil configuration GUI."""
    app = SigilApp()
    app.run()


if __name__ == "__main__":
    run_gui()
