# Gesture Replacement Summary

## Changes Made

### Replaced Gestures (Now Active)

#### Right Hand Gestures:
1. **arrow_left** (was: workspace_prev)
   - Hand: Right
   - Pose: Thumb_Down 👎
   - Action: `ydotool key 105:1 105:0` (Left arrow key)
   - Cooldown: 250ms

2. **arrow_right** (was: workspace_next)
   - Hand: Right  
   - Pose: Thumb_Up 👍
   - Action: `ydotool key 106:1 106:0` (Right arrow key)
   - Cooldown: 250ms

#### Left Hand Gestures:
3. **left_click** (was: launch_terminal)
   - Hand: Left
   - Pose: Closed_Fist ✊
   - Action: `ydotool click 0xC0` (Left mouse button)
   - Cooldown: 300ms

4. **right_click** (was: toggle_help)
   - Hand: Left
   - Pose: Victory ✌️
   - Action: `ydotool click 0xC1` (Right mouse button)
   - Cooldown: 300ms

### Removed Gestures

The following gestures were removed to make room for arrow keys and mouse clicks:
- ✗ `workspace_next` (Right Thumb_Up → workspace r+1)
- ✗ `workspace_prev` (Right Thumb_Down → workspace r-1)
- ✗ `launch_terminal` (Left Closed_Fist → kitty)
- ✗ `toggle_help` (Left Victory → sigil:toggle_help)

### Remaining Disabled Gestures

Two additional arrow key gestures remain disabled by default:
- `arrow_up` (Pointing_Up → Up arrow) - disabled
- `arrow_down` (Open_Palm → Down arrow) - disabled

These can be enabled if needed by changing `enabled: false` to `enabled: true`.

## Configuration Status

- **Total gestures**: 17
- **Enabled gestures**: 15
- **Disabled gestures**: 2 (arrow_up, arrow_down)

## Blanking Period

The config maintains a **1000ms (1 second) blanking period** after gesture activation:
- Prevents accidental double-activations
- Gives users time to reset hand position
- Configured in `execution.blanking_ms`

## Quick Reference

### Right Hand:
- ✊ Closed_Fist → Close window
- 🖐️ Open_Palm → Toggle launcher
- 👍 Thumb_Up → **Arrow RIGHT** ⬅️
- 👎 Thumb_Down → **Arrow LEFT** ➡️
- ☝️ Pointing_Up → Maximize
- ✌️ Victory → Toggle overview
- 🤟 ILoveYou → Lock session

### Left Hand:
- 🖐️ Open_Palm → Launch browser
- ✊ Closed_Fist → **Left CLICK** 🖱️
- ☝️ Pointing_Up → Media play/pause
- 👍 Thumb_Up → Media next
- 👎 Thumb_Down → Media previous
- ✌️ Victory → **Right CLICK** 🖱️

### Both Hands:
- Distance increase → Volume up
- Distance decrease → Volume down

## Testing

To test the new gestures:

```bash
# Test arrow keys manually
ydotool key 105:1 105:0  # Left arrow
ydotool key 106:1 106:0  # Right arrow

# Test mouse clicks manually  
ydotool click 0xC0  # Left click
ydotool click 0xC1  # Right click
```

## Notes

- All gestures are **enabled and active** by default
- ydotool must be running with proper permissions
- Gestures use instant type with low cooldown for responsive control
- Blanking period prevents gesture conflicts and accidental triggers
