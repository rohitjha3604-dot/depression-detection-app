# MiSec Brand Spec

Extracted from user brief: deep purple-to-indigo gradient used in the Streamlit app.

## Color tokens (bound to `:root`)

```css
:root {
  --bg:      #f7f6fc;   /* soft lavender page background */
  --surface: #ffffff;   /* cards, modals */
  --fg:      #1e1b2e;   /* deep purple-black text */
  --muted:   #6b6680;   /* secondary text */
  --border:  #e2dff0;   /* hairlines, dividers */
  --accent:  #667eea;   /* indigo primary — the bright end of the brand gradient */
}
```

Derived tones:
- `--accent-soft`: color-mix(in oklch, var(--accent) 14%, transparent)
- Hero background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) — used once, in the hero section only

## Typography

- Display: 'Iowan Old Style', 'Charter', Georgia, serif — warm, human, not clinical
- Body: system-ui stack — clean, readable
- Mono: ui-monospace stack — for stats, metadata, eyebrows

## Layout posture

- Generous whitespace — empathetic, calming
- One accent used at most twice per screen (eyebrow + primary CTA)
- The purple gradient is a single decisive flourish, confined to the hero background
- 10–16px radius on cards, hairline borders
- No shadows except subtle elevation on hero
