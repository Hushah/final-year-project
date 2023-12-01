import numpy as np


class SavedPose:
	# Constructor
	def __init__(self, hand_landmarks, input_key, image_width, image_height):
		# Attributes
		self.hand_landmarks = hand_landmarks
		self.input_key = input_key
		self.image_width = image_width
		self.image_height = image_height

		# The boundaries of the border
		self.min_width = None
		self.min_height = None
		self.max_width = None
		self.max_height = None

		# Relative position of the hands, in accordance with the border and not the larger canvas
		self.relative_hand_locations = np.empty((21, 2), dtype=float)

		# Calculating relative border and hand location coordinates within the border
		self.set_border_and_relative_locations()
		self.calculate_relative_locations()

	def get_hand_landmarks(self):
		return self.hand_landmarks

	def get_key(self):
		return self.input_key

	# Returning values in order: Min Width, Max Width, Min Height, Max Height
	def get_border(self, print_values=False):
		if print_values:
			print("BORDER:")
			print(f"Min Width: {self.min_width} \nMax Width: {self.max_width} "
			      f"\nMin Height: {self.min_height} \nMax Height: {self.max_height}")

		# return self.min_width, self.max_width, self.min_height, self.max_height

	# Function to set the relative border and populate relative_hand_locations array
	def set_border_and_relative_locations(self):
		index = 0
		# Loop through each landmark
		for landmarks in self.hand_landmarks:
			for landmark in landmarks.landmark:
				# DEBUG STATEMENTS
				'''
				# Print statements
				print(f"Landmark No: {index}")
				print(f"Landmark Name: {mp.solutions.hands.HandLandmark(index).name}")
				print(f"X Location: {landmark.x}")
				print(f"Y Location: {landmark.y}\n")
				'''

				# Populating array
				current_x = landmark.x * self.image_width
				current_y = landmark.y * self.image_height

				self.relative_hand_locations[index, 0] = current_x
				self.relative_hand_locations[index, 1] = current_y

				# If current landmark x is greater than self.max_width, assign it
				if self.max_width is None or current_x > self.max_width:
					self.max_width = current_x
				# If current landmark x is less than self.min_width, assign it
				if self.min_width is None or current_x < self.min_width:
					self.min_width = current_x
				# If current landmark y is greater than self.max_height, assign it
				if self.max_height is None or current_y > self.max_height:
					self.max_height = current_y
				# If current landmark y is less than self.min_height, assign it
				if self.min_height is None or current_y < self.min_height:
					self.min_height = current_y

				index += 1

	# Calculating the relative hand coordinates in accordance to the relative border
	def calculate_relative_locations(self):
		# Calculate for x
		self.relative_hand_locations[:, 0] -= self.min_width

		# Calculate for y
		self.relative_hand_locations[:, 1] -= self.min_height

	# Used to compare for equality of other objects
	def __hash__(self):
		return hash(self.input_key)

	# Used to compare for equality of other objects
	def __eq__(self, other):
		if isinstance(other, SavedPose):
			return self.input_key == other.input_key
		return False
