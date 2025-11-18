import cv2, time

for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
    print(f"\nTrying backend {backend}")
    for idx in range(0, 4):
        cap = cv2.VideoCapture(idx, backend)
        time.sleep(0.3)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        if ret:
            print(f"  ✅ Works: index {idx}")
        else:
            print(f"  ❌ Opened but no frame: index {idx}")
        cap.release()
