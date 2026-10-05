# MediaPipe pose models

This extension owns the lite, full and heavy pose landmarker model profiles.
Their official download locations and selection logic live in
`f8pymppose/runtime.py`.

Downloaded `.task` files belong in `${F8_MODEL_ROOT}/mediapipe`, outside the
extension installation. The SDK defaults this root to the platform user data
directory. Updating or uninstalling the extension leaves downloaded models intact.
