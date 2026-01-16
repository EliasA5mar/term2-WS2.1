import cv2
import numpy as np

def detect_blue_bottle():
    """
    Detects Solán de Cabras bottles (with blue color/label) in real-time using webcam feed.
    Press 'q' to quit, 'c' to adjust color range.
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Define range for blue color in HSV
    # You may need to adjust these values based on your bottle's specific blue shade
    lower_blue = np.array([100, 100, 50])   # Lower bound for blue
    upper_blue = np.array([130, 255, 255])  # Upper bound for blue

    print("Solán de Cabras Bottle Detection Started")
    print("Press 'q' to quit")
    print("Press 'c' to see current color range settings")
    print("\nLooking for blue Solán de Cabras bottles with characteristic rounded shape...")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame")
            break

        # Convert BGR to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for blue color
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and draw contours
        bottle_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter small contours (noise)
            if area > 500:  # Minimum area threshold

                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # === SOLÁN DE CABRAS BOTTLE SHAPE DETECTION ===
                # Check aspect ratio (Solán de Cabras is rounded, not as tall as typical bottles)
                aspect_ratio = h / float(w)

                # Calculate contour properties for shape analysis
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)  # More sensitive to curves
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)

                # Calculate solidity (how "filled" the shape is)
                solidity = area / hull_area if hull_area > 0 else 0

                # Calculate circularity (4π * area / perimeter²)
                # Solán de Cabras has a rounded shape, so circularity is important
                circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

                # Calculate extent (area / bounding box area)
                extent = area / (w * h) if (w * h) > 0 else 0

                # Check if upper portion is wider (characteristic teardrop shape)
                # Sample the contour at different heights
                contour_points = contour.reshape(-1, 2)
                upper_third = contour_points[contour_points[:, 1] < y + h/3]
                lower_third = contour_points[contour_points[:, 1] > y + 2*h/3]

                has_teardrop_shape = False
                if len(upper_third) > 0 and len(lower_third) > 0:
                    upper_width = np.max(upper_third[:, 0]) - np.min(upper_third[:, 0])
                    lower_width = np.max(lower_third[:, 0]) - np.min(lower_third[:, 0])
                    # Solán de Cabras is wider at the top/middle
                    has_teardrop_shape = upper_width >= lower_width * 0.8

                # Solán de Cabras bottle characteristics:
                # 1. Aspect ratio between 2.0 and 3.5 (rounded, not too tall)
                # 2. Solidity > 0.8 (very solid, smooth shape)
                # 3. Circularity > 0.4 (rounded shape, unlike straight bottles)
                # 4. Extent > 0.5 (fills bounding box well due to curves)
                # 5. Has teardrop/rounded characteristic
                # 6. More vertices due to curved shape (6-15 vertices)
                is_bottle = (2.0 <= aspect_ratio <= 3.5 and
                            solidity > 0.8 and
                            circularity > 0.4 and
                            extent > 0.5 and
                            has_teardrop_shape and
                            6 <= len(approx) <= 15)

                if is_bottle:
                    bottle_count += 1

                    # Draw rectangle around detected bottle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                    # Add label
                    cv2.putText(frame, f'Solan de Cabras {bottle_count}', (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Display shape metrics
                    cv2.putText(frame, f'AR: {aspect_ratio:.2f} | C: {circularity:.2f}', (x, y + h + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    # Draw red rectangle for rejected blue objects
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv2.putText(frame, f'Not Solan (AR:{aspect_ratio:.1f} C:{circularity:.2f})', (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Display detection count
        cv2.putText(frame, f'Bottles Detected: {bottle_count}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Show the frames
        cv2.imshow('Blue Bottle Detection', frame)
        cv2.imshow('Blue Mask', mask)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Exiting...")
            break
        elif key == ord('c'):
            print(f"Current blue range: Lower {lower_blue}, Upper {upper_blue}")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=" * 60)
    print("Solán de Cabras Bottle Detection System")
    print("=" * 60)
    print("\nMake sure you have the required libraries installed:")
    print("  pip install opencv-python numpy")
    print("\nNote: Adjust the HSV color range if detection is not accurate.")
    print("Common blue HSV ranges:")
    print("  Light blue: [90, 50, 50] to [110, 255, 255]")
    print("  Dark blue: [100, 100, 50] to [130, 255, 255]")
    print("\nSolán de Cabras Bottle Shape Requirements:")
    print("  - Aspect ratio: 2.0 to 3.5 (rounded, characteristic shape)")
    print("  - Solidity: > 0.8 (very solid, smooth contour)")
    print("  - Circularity: > 0.4 (rounded, not cylindrical)")
    print("  - Extent: > 0.5 (fills bounding box)")
    print("  - Teardrop shape: Wider at top/middle than bottom")
    print("  - Contour vertices: 6-15 (smooth curved edges)")
    print("\nColor coding:")
    print("  GREEN box (thick) = Solán de Cabras detected!")
    print("  RED box (thin) = Blue object, but wrong shape")
    print("  AR = Aspect Ratio | C = Circularity")
    print("=" * 60)
    print()

    detect_blue_bottle()
