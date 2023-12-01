# Import Statements
import cv2
import mediapipe as mp
import keyboard
import numpy as np
from SavedPoseClass import SavedPose

# Initialize MediaPipe hands Objects
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize webcam
cap = cv2.VideoCapture(0)

# Get the image width and height
image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# DEBUG STATEMENTS
# print("Image Width:", image_width)
# print("Image Height:", image_height)

# The most recent saved pose (SavedPoseClass Object)
saved_pose = None

# Set of saved poses (SavedPoseClass Objects)
poses_set = set()


# Function to compare the current hand pose on the webcam feed to the saved hand poses in poses_set
def compare_locations(current_hand_locations):
    # IMPORTANT NOTE: PLAY AROUND WITH THE TOLERANCE TO GET IT RIGHT. 50.0 seems like the sweet spot
    # my_rtol = 0
    my_atol = 50.0

    # If set of poses is not empty
    if len(poses_set) > 0:
        # Iterate through poses_set
        for pose in poses_set:
            # Compare whether the current hand pose on the webcam
            # is close in approximation to the current pose in poses_set
            comparison = np.isclose(current_hand_locations, pose.relative_hand_locations, atol=my_atol)
            # Check whether all values in comparison is True
            all_equal = np.all(comparison)

            # DEBUG STATEMENT
            # print(all_equal)

            # If all values in the comparison is True
            if all_equal:
                # Press and release the key for this pose in poses_set
                keyboard.press(pose.get_key())
                keyboard.release(pose.get_key())


# Initialize MediaPipe hands
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # While webcam is open
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Flip the image horizontally for a more intuitive mirror view
        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe hands
        results = hands.process(image)

        # Draw the hand landmarks on the image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # If hands are detected on screen
        if results.multi_hand_landmarks:
            # Assign input to key_pressed
            key_pressed = cv2.waitKey(1) & 0xFF

            # If the pressed key is not 255 (no key pressed), 27 (Escape key), or 0 (Arrow keys)
            if key_pressed not in (255, 27, 0):
                # Convert the ASCII value of the key to its corresponding character
                input_key = chr(key_pressed)

                # Create a new SavedPose instance with the hand landmarks and the input key
                saved_pose = SavedPose(hand_landmarks=results.multi_hand_landmarks, input_key=input_key,
                                       image_width=image_width, image_height=image_height)

                # If there is an object within the set that already has the same input_key as the new pose
                # We remove the existing element and replace it with the new one
                if saved_pose in poses_set:
                    poses_set.remove(saved_pose)

                poses_set.add(saved_pose)

            # For every hand landmark in the hand on screen
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand landmark on screen and the connections
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Create a 2d array for each hand location
                # Each element contains 2 elements, an x and a y coordinate
                # This array will be used to calculate relative hand locations for easy comparisons
                relative_hand_locations = np.empty((21, 2), dtype=float)
                index = 0
                # Iterate through each landmark within the current pose on screen
                for landmark in hand_landmarks.landmark:
                    # Save the hand location coordinate in the new array.
                    # Must multiply value by image width and height in order to get proper value
                    relative_hand_locations[index, 0] = landmark.x * image_width
                    relative_hand_locations[index, 1] = landmark.y * image_height

                    index += 1

                # Find the smallest and biggest x and y coordinates
                min_x = np.min(relative_hand_locations[:, 0])
                max_x = np.max(relative_hand_locations[:, 0])
                min_y = np.min(relative_hand_locations[:, 1])
                max_y = np.max(relative_hand_locations[:, 1])

                # Calculate relative position for x
                relative_hand_locations[:, 0] -= min_x
                # Calculate relative position for y
                relative_hand_locations[:, 1] -= min_y

                # DEBUG STATEMENT
                # print(relative_hand_locations)

                # Compare the hand locations of the current pose on screen with all the saved poses
                compare_locations(relative_hand_locations)

                # Draw the rectangle around the detected hand
                color = (0, 255, 0)  # Green
                thickness = 2
                cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), color, thickness)

        # Display the image
        cv2.imshow('main.py', image)

        # Exit the program when 'Esc' is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break


# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
