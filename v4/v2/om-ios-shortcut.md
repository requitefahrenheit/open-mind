# Open Mind — iOS Shortcuts Setup

Two Apple Shortcuts for sending content to Open Mind from your iPhone/iPad.

## Shortcut 1: "Add to Mind" (Share Sheet)

Captures shared text or URLs from any app via the share sheet.

### Steps to create

1. Open the **Shortcuts** app on your iPhone/iPad.
2. Tap **+** to create a new shortcut.
3. Tap the shortcut name at the top and rename it to **Add to Mind**.
4. Tap the **i** icon at the bottom, then enable **Show in Share Sheet**.
5. Under "Receives," select **Text**, **URLs**, and **Safari web pages**.
6. Add the following actions in order:

**Action 1 — Set Variable**
- Action: **Set Variable**
- Variable name: `input`
- Value: **Shortcut Input**

**Action 2 — Get URLs from Input (for URL extraction)**
- Action: **Get URLs from Input**
- Input: `input`

**Action 3 — If**
- Action: **If**
- Condition: **URLs** *has any value*

**Action 4 (inside If) — Set Variable**
- Action: **Set Variable**
- Variable name: `payload`
- Value: **URLs**

**Action 5 — Otherwise**

**Action 6 (inside Otherwise) — Set Variable**
- Action: **Set Variable**
- Variable name: `payload`
- Value: `input`

**Action 7 — End If**

**Action 8 — Get Contents of URL (POST request)**
- Action: **Get Contents of URL**
- URL: `https://openmind.fahrenheitrequited.dev/api/nl`
- Method: **POST**
- Headers: `Content-Type` = `application/json`
- Request Body: **JSON**
  - `text` = `payload`
  - `channel` = `ios`

**Action 9 — Get Dictionary Value**
- Action: **Get Dictionary Value**
- Key: `response`
- Dictionary: output of previous action

**Action 10 — Show Notification**
- Action: **Show Notification**
- Title: **Open Mind**
- Body: output of previous action

### Simplified alternative

If the above is too many steps, you can use a simpler version:

1. Create shortcut, name it **Add to Mind**, enable Share Sheet.
2. Add **Get Contents of URL**:
   - URL: `https://openmind.fahrenheitrequited.dev/api/nl`
   - Method: POST
   - Headers: `Content-Type` = `application/json`
   - Request Body (JSON): `text` = **Shortcut Input**, `channel` = `ios`
3. Add **Get Dictionary Value** with key `response`.
4. Add **Show Notification** with the result.

---

## Shortcut 2: "Ask Mind" (Standalone)

Prompts for text input, then sends it to Open Mind. Use from the home screen or Siri.

### Steps to create

1. Open the **Shortcuts** app.
2. Tap **+** to create a new shortcut.
3. Rename it to **Ask Mind**.
4. Add the following actions:

**Action 1 — Ask for Input**
- Action: **Ask for Input**
- Prompt: `What's on your mind?`
- Input Type: **Text**

**Action 2 — Get Contents of URL (POST request)**
- Action: **Get Contents of URL**
- URL: `https://openmind.fahrenheitrequited.dev/api/nl`
- Method: **POST**
- Headers: `Content-Type` = `application/json`
- Request Body: **JSON**
  - `text` = output of **Ask for Input**
  - `channel` = `ios`

**Action 3 — Get Dictionary Value**
- Action: **Get Dictionary Value**
- Key: `response`
- Dictionary: output of previous action

**Action 4 — Show Result**
- Action: **Show Result**
- Text: output of previous action

### Optional: Add to Home Screen

After creating either shortcut, tap the **i** icon and select **Add to Home Screen** for one-tap access.

### Optional: Siri Trigger

Both shortcuts are automatically available via Siri. Just say:
- "Hey Siri, Add to Mind"
- "Hey Siri, Ask Mind"

---

## API Reference

Both shortcuts hit the same endpoint:

```
POST https://openmind.fahrenheitrequited.dev/api/nl
Content-Type: application/json

{
  "text": "your input here",
  "channel": "ios"
}
```

Response:

```json
{
  "action": "add",
  "node": { "id": "...", "label": "..." },
  "response": "Added: your node label"
}
```

The `response` field always contains a human-readable confirmation or search results summary.
