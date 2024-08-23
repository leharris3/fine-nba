### TODO
1. Extract all time-remaining values from the `filtered-clips` dataset.
2. Align statvu `moments` to annotations via time-remaining
3. Verify results (quantitatively + qualitatively)

### Time-Remaining Extraction
- Two simple steps
	1. localize roi
		- We already have a yolo model of the form (img -> roi)
	2. detect text
		- Find something fast + performant off-the-shelf
	3. recognize text
		- Trickier. Off-the-shelf models are pretty bad. Probably the easiest thing to do is ft a fast/accurate text-rec model.
		- We want to include global context. That is, don't make character-level predictions in isolation. Use a transformer-based approach.
