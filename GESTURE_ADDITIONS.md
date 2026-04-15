# Gesture Additions Summary

## Changes Made

### 1. Blanking Period Documentation
- Added explanatory comment to `execution.blanking_ms` setting in `config.yaml`
- The existing 1000ms blanking period already provides a "blank period after gesture activates"
- This prevents accidental double-activations and gives users time to reset their hand position

### 2. Arrow Key Gestures (ydotool)
Added 4 new gesture mappings for arrow key control:

- **arrow_up**: `Pointing_Up` pose → Up arrow key
- **arrow_down**: `Open_Palm` pose → Down arrow key  
- **arrow_left**: `Thumb_Down` pose → Left arrow key
- **arrow_right**: `Thumb_Up` pose → Right arrow key

All set to `hand: both` (trigger on either hand) with 250ms cooldown.

### 3. Mouse Click Gestures (ydotool)
Added 2 new gesture mappings for mouse control:

- **left_click**: `Closed_Fist` pose → Left mouse button
- **right_click**: `Victory` pose → Right mouse button

Both set to `hand: both` (trigger on either hand) with 300ms cooldown.

## Important Notes

### Gestures Are Disabled By Default
All 6 new gestures are set to `enabled: false` to avoid conflicts with existing gestures.

**Why?** Many of the poses (Pointing_Up, Thumb_Up, Thumb_Down, Victory, Closed_Fist, Open_Palm) 
are already used by single-hand gestures for window management and media control.

### How to Enable

To use these gestures:

1. **Option A - Enable the new gestures:**
   - In `config.yaml`, change `enabled: false` to `enabled: true` for desired gestures
   - Note: They will trigger on EITHER hand making the pose

2. **Option B - Disable conflicting gestures:**
   - Set `enabled: false` on existing conflicting gestures (right/left hand specific ones)
   - Then enable the new arrow/mouse gestures

### Conflicting Gestures

| New Gesture | Conflicts With |
|------------|----------------|
| arrow_up (Pointing_Up) | toggle_maximize (right), media_play_pause (left) |
| arrow_down (Open_Palm) | toggle_launcher (right), launch_browser (left) |
| arrow_left (Thumb_Down) | workspace_prev (right), media_prev (left) |
| arrow_right (Thumb_Up) | workspace_next (right), media_next (left) |
| left_click (Closed_Fist) | close_window (right), launch_terminal (left) |
| right_click (Victory) | toggle_overview (right), toggle_help (left) |

## Technical Details

### ydotool Key Codes
- Up arrow: 103
- Down arrow: 108
- Left arrow: 105
- Right arrow: 106

### ydotool Mouse Codes
- Left click: 0xC0
- Right click: 0xC1

### Command Format
All commands are executed through hyprctl dispatch:
```yaml
action: "hyprctl dispatch exec ydotool key 103:1 103:0"  # Press and release
action: "hyprctl dispatch exec ydotool click 0xC0"       # Mouse click
```

## Testing

To test if ydotool is working:
```bash
# Test arrow key
ydotool key 103:1 103:0  # Should send Up arrow

# Test mouse click
ydotool click 0xC0  # Should left-click
```

Note: ydotool may require proper permissions/daemon running.
